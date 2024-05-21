import os
import glob
import torch 
from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt
from transformers4rec.torch.utils.examples_utils import wipe_memory
from merlin_standard_lib.schema.schema import Schema
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

INPUT_DATA_DIR = "/home/vmagent/app/recsys2022_transformer/data_feat/"
out_dir = "/home/vmagent/app/recsys2022_transformer/data_feat/models/base"
num_train_epochs = 10
train_paths = glob.glob(os.path.join(INPUT_DATA_DIR, f"train_1000.parquet"))
eval_paths = glob.glob(os.path.join(INPUT_DATA_DIR, f"valid_100.parquet"))
# train_paths = glob.glob(os.path.join(INPUT_DATA_DIR, f"train.parquet"))
# eval_paths = glob.glob(os.path.join(INPUT_DATA_DIR, f"valid.parquet"))
print(train_paths)

per_device_train_batch_size = 2
per_device_eval_batch_size = 128
schema = Schema().from_proto_text(path_or_proto_text=os.path.join(INPUT_DATA_DIR, "schema.pbtxt"))
schema = schema.select_by_name(['item_id-list',"wf"])
# schema = schema.select_by_name(['item_id-list',"f1-list", "f2-list", "f3-list"])
# print(schema)

inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=101,
        continuous_projection=64,
        masking="clm",
        d_output=100,
        train_on_last_item_seq_only=True
)

transformer_config = tr.XLNetConfig.build(
    d_model=64, n_head=8, n_layer=2, total_seq_length=101
)
body = tr.SequentialBlock(
    inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
)
metrics = [NDCGAt(top_ks=[20, 40], labels_onehot=True),  
           RecallAt(top_ks=[20, 40], labels_onehot=True)]

head = tr.Head(
    body,
    tr.NextItemPredictionTask(target_dim=28144,
                                weight_tying=True, 
                                metrics=metrics),
    inputs=inputs,
)

model = tr.Model(head).cuda()
# print(model)

train_args = T4RecTrainingArguments(data_loader_engine='merlin',
                                    dataloader_drop_last = False,
                                    gradient_accumulation_steps = 1,
                                    per_device_train_batch_size = per_device_train_batch_size, 
                                    per_device_eval_batch_size = per_device_eval_batch_size,
                                    output_dir = out_dir, 
                                    learning_rate=0.0005,
                                    lr_scheduler_type='cosine', 
                                    learning_rate_num_cosine_cycles_by_epoch=1.5,
                                    num_train_epochs=num_train_epochs,
                                    max_sequence_length=101, 
                                    report_to = [],
                                    logging_steps=500,
                                    no_cuda=False,
                                    predict_top_k=100,
                                    evaluation_strategy="epoch",
                                    save_strategy="epoch")

# Instantiate the T4Rec Trainer, which manages training and evaluation for the PyTorch API
trainer = Trainer(
    model=model,
    args=train_args,
    schema=schema,
    compute_metrics=True,
)

trainer.train_dataset_or_path = train_paths
trainer.eval_dataset_or_path = eval_paths
trainer.reset_lr_scheduler()
trainer.train()
trainer.state.global_step +=1
print('train finished')

preds = trainer.evaluate(metric_key_prefix='eval')

predictions = torch.Tensor(preds.predictions[0]).int()
labels = torch.Tensor(preds.label_ids).int().unsqueeze(-1)
num_samples = labels.shape[0]
    
hit_ranks = torch.where(predictions == labels)[1] + 1
hit = hit_ranks.numel()
mrr = hit_ranks.float().reciprocal().sum().item()
print(f"mrr is {(mrr * 1.0 /num_samples):.5f}")
print(f"hit is {(hit * 1.0 /num_samples):.5f}")