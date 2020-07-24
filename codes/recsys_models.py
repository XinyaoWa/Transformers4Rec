
import logging

import torch.nn as nn

# load transformer model and its configuration classes
from transformers.modeling_xlnet import XLNetModel
from transformers.configuration_xlnet import XLNetConfig
from transformers.modeling_gpt2 import GPT2Model
from transformers.configuration_gpt2 import GPT2Config
from transformers.modeling_longformer import LongformerModel
from transformers.configuration_longformer import LongformerConfig
from transformers.configuration_utils import PretrainedConfig

from models.gru4rec import GRU4REC

logger = logging.getLogger(__name__)


def get_recsys_model(model_args, data_args, training_args, target_size=None):

    if model_args.model_type == 'xlnet':
        model_cls = XLNetModel
        config = XLNetConfig(
            d_model=model_args.d_model,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            d_inner=model_args.d_model * 4,
            ff_activation=model_args.hidden_act,
            untie_r=True,
            attn_type="uni",
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
        )

    #NOTE: gpt2 and longformer are not fully tested supported yet.

    elif model_args.model_type == 'gpt2':
        model_cls = GPT2Model
        config = GPT2Config(
            n_embd=model_args.d_model,
            n_layer=model_args.n_layer,
            n_head=model_args.n_head,
            activation_function=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            n_positions=data_args.max_seq_len,
        )

    elif model_args.model_type == 'longformer':
        model_cls = LongformerModel
        config = LongformerConfig(
            hidden_size=model_args.d_model,
            num_hidden_layers=model_args.n_layer,
            num_attention_heads=model_args.n_head,
            hidden_act=model_args.hidden_act,
            initializer_range=model_args.initializer_range,
            layer_norm_eps=model_args.layer_norm_eps,
            dropout=model_args.dropout,
            max_position_embeddings=data_args.max_seq_len,
            vocab_size=target_size,
        )

    elif model_args.model_type == 'gru':
        model_cls = nn.GRU(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig()  # dummy config

    elif model_args.model_type == 'lstm':
        model_cls = nn.LSTM(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig()

    elif model_args.model_type == 'rnn':
        model_cls = nn.RNN(
            input_size=model_args.d_model,
            num_layers=model_args.n_layer,
            hidden_size=model_args.d_model,
            dropout=model_args.dropout,
        )
        config = PretrainedConfig()

    elif model_args.model_type == 'gru4rec':
        model_cls = GRU4REC(
            input_size=model_args.d_model, 
            hidden_size=model_args.d_model, 
            output_size=model_args.d_model, 
            num_layers=model_args.n_layer, 
            final_act='tanh',
            dropout_hidden=model_args.dropout, 
            dropout_input=model_args.dropout, 
            batch_size=training_args.per_device_train_batch_size, 
            embedding_dim=-1, 
            use_cuda=False
        )

    else:
        raise NotImplementedError

    if model_args.model_type in ['gru', 'lstm', 'gru4rec']:
        model = model_cls

    elif model_args.model_name_or_path:
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_cls(config)
        
    return model, config