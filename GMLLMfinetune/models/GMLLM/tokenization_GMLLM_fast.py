# coding=utf-8
from transformers import RobertaTokenizerFast, XLMRobertaTokenizerFast
from transformers.file_utils import is_sentencepiece_available
from transformers.utils import logging


if is_sentencepiece_available():
    from .tokenization_GMLLM import GMLLMTokenizer
else:
    GMLLMTokenizer = None


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

with open('tag.txt', 'r') as tagf:
    TAG = tagf.read().lower()
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'

if TAG == 'monolingual':
    class GMLLMTokenizerFast(RobertaTokenizerFast):

        vocab_files_names = VOCAB_FILES_NAMES
        max_model_input_sizes = {"gmllm-roberta-base": 512,}
        model_input_names = ["input_ids", "attention_mask"]
        slow_tokenizer_class = GMLLMTokenizer

        def __init__(self, model_max_length=512, **kwargs):
            super().__init__(model_max_length=model_max_length, **kwargs)
            
elif TAG == 'multilingual':
    class GMLLMTokenizerFast(XLMRobertaTokenizerFast):

        vocab_files_names = VOCAB_FILES_NAMES
        max_model_input_sizes = {"gmllm-infoxlm-base": 512,}
        model_input_names = ["input_ids", "attention_mask"]
        slow_tokenizer_class = GMLLMTokenizer

        def __init__(self, model_max_length=512, **kwargs):
            super().__init__(model_max_length=model_max_length, **kwargs)
