from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, RobertaConverter, XLMRobertaConverter
from transformers.models.auto.modeling_auto import _BaseAutoModelClass, auto_class_update, AutoModelForTokenClassification
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES, _LazyConfigMapping
from transformers.models.auto.auto_factory import _LazyAutoMapping

from .models.GMLLM import (
    GMLLMConfig,
    GMLLMForRelationExtraction,
    GMLLMForTokenClassification,
    GMLLMTokenizer,
    GMLLMTokenizerFast,
)

CONFIG_MAPPING.register("gmllm", GMLLMConfig)
MODEL_NAMES_MAPPING.update([("gmllm", "GMLLM"),])
TOKENIZER_MAPPING.register(GMLLMConfig, (GMLLMTokenizer, GMLLMTokenizerFast))

with open('tag.txt', 'r') as tagf:
    TAG = tagf.read().lower()
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'
if TAG == 'monolingual':
    SLOW_TO_FAST_CONVERTERS.update({"GMLLMTokenizer": RobertaConverter,})
elif TAG == 'multilingual':
    SLOW_TO_FAST_CONVERTERS.update({"GMLLMTokenizer": XLMRobertaConverter,})

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.register(GMLLMConfig, GMLLMForTokenClassification)
AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")

# add relation extraction auto model
MODEL_FOR_RELATION_EXTRACTION_MAPPING_NAMES = OrderedDict(
    [
        ("gmllm", "GMLLMForRelationExtraction"),
    ]
)

CONFIG_GMLLM_MAPPING_NAMES = OrderedDict(
    [
        #("gmllm", "gmllm"),
    ]
)
MODEL_FOR_RELATION_EXTRACTION_MAPPING = _LazyAutoMapping(
    CONFIG_GMLLM_MAPPING_NAMES, MODEL_FOR_RELATION_EXTRACTION_MAPPING_NAMES
)

MODEL_FOR_RELATION_EXTRACTION_MAPPING.register(GMLLMConfig, GMLLMForRelationExtraction)

class AutoModelForRelationExtraction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_RELATION_EXTRACTION_MAPPING

AutoModelForRelationExtraction = auto_class_update(AutoModelForRelationExtraction, head_doc="relation extraction")
