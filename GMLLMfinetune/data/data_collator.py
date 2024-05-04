from dataclasses import dataclass
from typing import Optional, Union

import torch

#from detectron2.structures import ImageList
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        has_image_input = "image" in features[0]
        if has_image_input:
            #image = ImageList.from_tensors([torch.tensor(feature["image"]) for feature in features], 32)
            for feature in features:
                del feature["image"]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        if 'position_ids' in batch:
            token_length = len(batch["input_ids"][0])
            batch["position_ids"] = [pos_ids + [1] * (token_length - len(pos_ids)) for pos_ids in batch["position_ids"]]
        if 'wbbox' in batch:
            pos_length = len(batch["wbbox"][0][0])
            wbox_length = max([len(pos) for pos in batch["wbbox"]])
            batch["wbbox_mask"] = [[1] * len(bbox) + [0] * (wbox_length - len(bbox)) for bbox in batch["wbbox"]]
            batch["wbbox"] = [bbox + [[0] * pos_length] * (wbox_length - len(bbox)) for bbox in batch["wbbox"]]
            batch["layout_windex"] = [index + [0] * (sequence_length - len(index)) for index in batch["layout_windex"]]
        if 'lbbox' in batch:
            pos_length = len(batch["lbbox"][0][0])
            lbox_length = max([len(pos) for pos in batch["lbbox"]])
            batch["lbbox_mask"] = [[1] * len(bbox) + [0] * (lbox_length - len(bbox)) for bbox in batch["lbbox"]]
            batch["lbbox"] = [bbox + [[0] * pos_length] * (lbox_length - len(bbox)) for bbox in batch["lbbox"]]
            batch["layout_lindex"] = [index + [0] * (sequence_length - len(index)) for index in batch["layout_lindex"]]
            if 'wbbox' in batch:
                batch["layout_wlindex"] = [index + [0] * (wbox_length - len(index)) for index in batch["layout_wlindex"]]
        if 'rbbox' in batch:
            pos_length = len(batch["rbbox"][0][0])
            rbox_length = max([len(pos) for pos in batch["rbbox"]])
            batch["rbbox_mask"] = [[1] * len(bbox) + [0] * (rbox_length - len(bbox)) for bbox in batch["rbbox"]]
            batch["rbbox"] = [bbox + [[0] * pos_length] * (rbox_length - len(bbox)) for bbox in batch["rbbox"]]
            batch["layout_rindex"] = [index + [0] * (sequence_length - len(index)) for index in batch["layout_rindex"]]
            if 'lbbox' in batch:
                batch["layout_lrindex"] = [index + [0] * (lbox_length - len(index)) for index in batch["layout_lrindex"]]
        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        
        return batch
