import os
import glob
import torch 
from transformers4rec import torch as tr
from transformers4rec.torch.ranking_metric import NDCGAt, AvgPrecisionAt, RecallAt
from transformers4rec.torch.utils.examples_utils import wipe_memory
from merlin_standard_lib.schema.schema import Schema
from transformers4rec.config.trainer import T4RecTrainingArguments
from transformers4rec.torch import Trainer

schema = Schema().from_proto_text(path_or_proto_text="data_nv/schema.pbtxt")
schema = schema.select_by_name(["item_id-list","day-first"])
print(schema)

inputs = tr.TabularSequenceFeatures.from_schema(
        schema,
        max_sequence_length=20,
        # continuous_projection=64,
        masking="mlm",
        d_output=100,
)

# Define XLNetConfig class and set default parameters for HF XLNet config  
transformer_config = tr.XLNetConfig.build(
    d_model=64, n_head=4, n_layer=2, total_seq_length=20
)
# Define the model block including: inputs, masking, projection and transformer block.
body = tr.SequentialBlock(
    inputs, tr.MLPBlock([64]), tr.TransformerBlock(transformer_config, masking=inputs.masking)
)

# Define the evaluation top-N metrics and the cut-offs
metrics = [NDCGAt(top_ks=[20, 40], labels_onehot=True),  
           RecallAt(top_ks=[20, 40], labels_onehot=True)]

# Define a head related to next item prediction task 
head = tr.Head(
    body,
    tr.NextItemPredictionTask(weight_tying=True, 
                              metrics=metrics),
    inputs=inputs,
)

# Get the end-to-end Model class 
model = tr.Model(head)

# Set hyperparameters for training 
train_args = T4RecTrainingArguments(data_loader_engine='merlin', 
                                    dataloader_drop_last = True,
                                    gradient_accumulation_steps = 1,
                                    per_device_train_batch_size = 128, 
                                    per_device_eval_batch_size = 32,
                                    output_dir = "./tmp", 
                                    learning_rate=0.0005,
                                    lr_scheduler_type='cosine', 
                                    learning_rate_num_cosine_cycles_by_epoch=1.5,
                                    num_train_epochs=5,
                                    max_sequence_length=20, 
                                    report_to = [],
                                    logging_steps=50,
                                    no_cuda=True)

# Instantiate the T4Rec Trainer, which manages training and evaluation for the PyTorch API
trainer = Trainer(
    model=model,
    args=train_args,
    schema=schema,
    compute_metrics=True,
)

train_paths = glob.glob("data_nv/part_0.parquet")

trainer.train_dataset_or_path = train_paths
trainer.reset_lr_scheduler()
trainer.train()
trainer.state.global_step +=1
print('finished')

wipe_memory()