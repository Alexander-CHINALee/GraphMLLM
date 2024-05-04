# coding=utf-8
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PretrainedConfig, PreTrainedTokenizer, TensorType

from transformers.utils import logging
from transformers import RobertaConfig, XLMRobertaConfig

logger = logging.get_logger(__name__)

with open('tag.txt', 'r') as tagf:
    TAG = tagf.read().lower()
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'

if TAG == 'monolingual':
    class GMLLMConfig(RobertaConfig):
        model_type = "gmllm"

        def __init__(
            self,
            max_2d_position_embeddings=1024,
            layout_type='word_line',
            layout_params='100,50',
            num_layout_attention_heads=12,
            layout_hidden_size=128,
            layout_intermediate_size=768,
            **kwargs
        ):
            super().__init__(
                **kwargs,
            )
            self.max_2d_position_embeddings = max_2d_position_embeddings
            self.layout_type = layout_type
            self.layout_params = layout_params
            self.num_layout_attention_heads = num_layout_attention_heads
            self.layout_hidden_size = layout_hidden_size
            self.layout_intermediate_size = layout_intermediate_size

elif TAG == 'multilingual':
    class GMLLMConfig(XLMRobertaConfig):
        model_type = "gmllm"

        def __init__(
            self,
            max_2d_position_embeddings=1024,
            layout_type='word_line',
            layout_params='100,50',
            num_layout_attention_heads=12,
            layout_hidden_size=128,
            layout_intermediate_size=768,
            **kwargs
        ):
            super().__init__(
                **kwargs,
            )
            self.max_2d_position_embeddings = max_2d_position_embeddings
            self.layout_type = layout_type
            self.layout_params = layout_params
            self.num_layout_attention_heads = num_layout_attention_heads
            self.layout_hidden_size = layout_hidden_size
            self.layout_intermediate_size = layout_intermediate_size

