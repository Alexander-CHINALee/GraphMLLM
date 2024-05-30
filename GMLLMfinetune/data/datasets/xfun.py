# Lint as: python3
import json
import logging
import os
import numpy as np
import datasets

from GMLLMfinetune.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer
from .XYCut import augment_xy_cut


_URL = "https://github.com/doc-analysis/XFUN/releases/download/v1.0/"

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)

import cv2
def visual(h, w, bboxes, color_idx, output):
    img = np.zeros((h, w, 3), dtype=np.int16)
    color = [(255,0,0),(0,255,0)]
    for box, idx in zip(bboxes, color_idx):
        x1,y1,x2,y2 = box
        h, w = y2 - y1, x2 - x1
        cv2.rectangle(img, (x1, y1), (x2, y2), color[idx], thickness=2)
    cv2.imwrite(output, img)

def text_region(bboxes):
    layout_index = [0 for _ in range(len(bboxes))]
    reg_x1,reg_y1,reg_x2,reg_y2 = 10000, 10000, 0, 0
    for bbox in bboxes:
        box_x1,box_y1,box_x2,box_y2 = min(bbox[0::2]),min(bbox[1::2]),max(bbox[0::2]),max(bbox[1::2])
        reg_x1,reg_y1,reg_x2,reg_y2 = min(reg_x1, box_x1),min(reg_y1, box_y1),max(reg_x2, box_x2),max(reg_y2, box_y2)
    region_bboxes = [[reg_x1,reg_y1,reg_x2,reg_y2]]
    return region_bboxes, layout_index

class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""

    BUILDER_CONFIGS = [XFUNConfig(name=f"xfun.{lang}", lang=lang) for lang in _LANG]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "position_ids": datasets.Sequence(datasets.Value("int64")),
                    "wbbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "lbbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "rbbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "layout_windex": datasets.Sequence(datasets.Value("int64")),
                    "layout_lindex": datasets.Sequence(datasets.Value("int64")),
                    "layout_wlindex": datasets.Sequence(datasets.Value("int64")),
                    "layout_rindex": datasets.Sequence(datasets.Value("int64")),
                    "layout_lrindex": datasets.Sequence(datasets.Value("int64")),
                    "labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=["O", "B-QUESTION", "B-ANSWER", "B-HEADER", "I-ANSWER", "I-QUESTION", "I-HEADER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=["HEADER", "QUESTION", "ANSWER"]),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "start_index": datasets.Value("int64"),
                            "end_index": datasets.Value("int64"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": [f"{_URL}{self.config.lang}.train.json", f"{_URL}{self.config.lang}.train.zip"],
            "val": [f"{_URL}{self.config.lang}.val.json", f"{_URL}{self.config.lang}.val.zip"],
            # "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]
        # test_files_for_many_langs = [downloaded_files["test"]]
        if self.config.additional_langs:
            additional_langs = self.config.additional_langs.split("+")
            if "all" in additional_langs:
                additional_langs = [lang for lang in _LANG if lang != self.config.lang]
            for lang in additional_langs:
                urls_to_download = {"train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]}
                additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
                train_files_for_many_langs.append(additional_downloaded_files["train"])

        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}
            ),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r") as f:
                data = json.load(f)
            for doc in data["documents"]:
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                image, size = load_image(doc["img"]["fpath"])
                document = doc["document"]
                tokenized_doc = {"input_ids": [], "wbbox": [], "lbbox": [], "layout_windex":[], "layout_lindex":[],"layout_wlindex":[], "labels": []}
                entities = []
                relations = []
                id2label = {}
                entity_id_to_index_map = {}
                empty_entity = set()
                for line in document:
                    #for line in ordered_forms:
                    if len(line["text"]) == 0:
                        empty_entity.add(line["id"])
                        continue
                    id2label[line["id"]] = line["label"]
                    relations.extend([tuple(sorted(l)) for l in line["linking"]])
                    if '/en' in filepath[0]:
                        tokenized_inputs = self.tokenizer(
                            ' '.join([q['text'].replace(u'\uf703','') for q in line['words']]),
                            add_special_tokens=False,
                            return_offsets_mapping=True,
                            return_attention_mask=False,
                        )
                    else:
                        tokenized_inputs = self.tokenizer(
                            line["text"],
                            add_special_tokens=False,
                            return_offsets_mapping=True,
                            return_attention_mask=False,
                        )
                    text_length = 0
                    ocr_length = 0
                    word_bboxes = []
                    line_bboxes = []
                    word_index = []
                    line_index = []
                    word_line_index = []
                    line_bbox = normalize_bbox(simplify_bbox(line["box"]), size)
                    word_bbox_off = len(tokenized_doc["wbbox"])
                    add_word_index = False
                    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                        line_index.append(len(tokenized_doc["lbbox"]))
                        if token_id == 6:
                            if line["words"]:
                                add_word_index = True
                            else:
                                word_index.append(word_bbox_off + len(word_bboxes)-1)
                            continue
                        text_length += offset[1] - offset[0]
                        tmp_box = []
                        while ocr_length < text_length:
                            ocr_word = line["words"].pop(0)
                            ocr_length += len(
                                self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                            )
                            tmp_box.append(simplify_bbox(ocr_word["box"]))
                        if len(tmp_box) != 0:
                            word_bboxes.append(normalize_bbox(merge_bbox(tmp_box), size))
                            word_line_index.append(len(tokenized_doc["lbbox"]))
                        if add_word_index:
                            word_index.append(word_bbox_off + len(word_bboxes)-1)
                            add_word_index = False
                        word_index.append(word_bbox_off + len(word_bboxes)-1)
                    
                    # line bbox
                    x1,y1,x2,y2 = line_bbox
                    center_bbox = [x1,y1,x2,y2]
                    line_bboxes.append(center_bbox)
                    
                    if line["label"] == "other":
                        label = ["O"] * len(line_index)
                    else:
                        label = [f"I-{line['label'].upper()}"] * len(line_index)
                        label[0] = f"B-{line['label'].upper()}"
                    tokenized_inputs.update({"wbbox": word_bboxes, "lbbox": line_bboxes, "layout_windex":word_index, \
                                             "layout_lindex":line_index, "layout_wlindex":word_line_index, "labels": label})
                    if label[0] != "O":
                        entity_id_to_index_map[line["id"]] = len(entities)
                        entities.append(
                            {
                                "start": len(tokenized_doc["input_ids"]),
                                "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                                "label": line["label"].upper(),
                            }
                        )
                    for i in tokenized_doc:
                        tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
                # reorder
                line_word_group = [[] for _ in range(len(tokenized_doc['lbbox']))]
                for word_i, line_i in enumerate(tokenized_doc['layout_wlindex']):
                    line_word_group[line_i].append(word_i)
                orders = augment_xy_cut(tokenized_doc['lbbox'],direction='y')
                wpos_ids = []
                for line_i in orders:
                    wpos_ids += line_word_group[line_i]
                
                word_token_group = [[] for _ in range(len(wpos_ids))]
                for token_i,word_i in enumerate(tokenized_doc['layout_windex']):
                    word_token_group[word_i].append(token_i)
                token_ids = []
                for word_i in wpos_ids:
                    token_ids += word_token_group[word_i]
                position_ids = [-1 for _ in range(len(token_ids))]
                pos_i = 0
                for token_i in token_ids:
                    position_ids[token_i] = pos_i
                    pos_i += 1
                tokenized_doc['position_ids'] = position_ids
                assert len(position_ids) == len(tokenized_doc['input_ids']) and -1 not in position_ids
                relations = list(set(relations))
                relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
                kvrelations = []
                for rel in relations:
                    pair = [id2label[rel[0]], id2label[rel[1]]]
                    if pair == ["question", "answer"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                        )
                    elif pair == ["answer", "question"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                        )
                    else:
                        continue

                def get_relation_span(rel):
                    bound = []
                    for entity_index in [rel["head"], rel["tail"]]:
                        bound.append(entities[entity_index]["start"])
                        bound.append(entities[entity_index]["end"])
                    return min(bound), max(bound)

                relations = sorted(
                    [
                        {
                            "head": rel["head"],
                            "tail": rel["tail"],
                            "start_index": get_relation_span(rel)[0],
                            "end_index": get_relation_span(rel)[1],
                        }
                        for rel in kvrelations
                    ],
                    key=lambda x: x["head"],
                )
                chunk_size = 510
                for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                    item = {}
                    for k in tokenized_doc:
                        if k == 'wbbox' or k == 'lbbox' or k == 'rbbox' or k == 'layout_wlindex':
                            continue
                        else:
                            item[k] = tokenized_doc[k][index : index + chunk_size]
                    end_index = min(index + chunk_size, len(tokenized_doc["layout_lindex"]))
                    
                    word_idx_s,word_idx_e = tokenized_doc["layout_windex"][index], tokenized_doc["layout_windex"][end_index-1] + 1
                    item['wbbox'] = tokenized_doc['wbbox'][word_idx_s : word_idx_e]
                    item['layout_windex'] = [lay_idx - word_idx_s for lay_idx in item['layout_windex']]
                    
                    line_idx_s,line_idx_e = tokenized_doc["layout_lindex"][index], tokenized_doc["layout_lindex"][end_index-1] + 1
                    item['lbbox'] = tokenized_doc['lbbox'][line_idx_s : line_idx_e]
                    item['layout_lindex'] = [lay_idx - line_idx_s for lay_idx in item['layout_lindex']]
                    item['layout_wlindex'] = [lay_idx - line_idx_s for lay_idx in tokenized_doc['layout_wlindex'][word_idx_s : word_idx_e]]
                    
                    region_bbox, region_index = text_region(item['lbbox'])
                    item['rbbox'] = [[0, 0, 0, 0]] + region_bbox + [[1000, 1000, 1000, 1000]]
                    item['layout_rindex'] = [0] + [region_index[idx] + 1 for idx in item['layout_lindex']] + [len(item['rbbox'])-1]
                    item['layout_lrindex'] = [0] + [idx + 1 for idx in region_index] + [len(item['rbbox'])-1]
                    add_num = 1
                    # add cls and end tokens
                    item['input_ids'] = [self.tokenizer.cls_token_id] + item['input_ids'] + [self.tokenizer.sep_token_id]
                    # position_ids
                    sort_pos_ids = sorted(enumerate(item['position_ids']),key = lambda x:x[1])
                    start_ids = 0
                    new_position_ids = [-1] * len(sort_pos_ids)
                    for pos_item in sort_pos_ids:
                        new_position_ids[pos_item[0]] = start_ids
                        start_ids += 1
                    assert -1 not in new_position_ids
                    item['position_ids'] = [2] + [ids + 3 for ids in new_position_ids] + [len(new_position_ids)+3]
                    item['labels'] = ['O'] + item['labels'] + ['O']
                    item['wbbox'] = [[0,0,0,0]] + item['wbbox'] + [[1000,1000,1000,1000]]
                    item['layout_windex'] = [0] + [idx + 1 for idx in item['layout_windex']] + [len(item['wbbox'])-1]
                    item['lbbox'] = [[0,0,0,0]] + item['lbbox'] + [[1000,1000,1000,1000]]
                    item['layout_lindex'] = [0] + [idx + 1 for idx in item['layout_lindex']] + [len(item['lbbox'])-1]
                    item['layout_wlindex'] = [0] + [idx + 1 for idx in item['layout_wlindex']] + [len(item['lbbox'])-1]
                    entities_in_this_span = []
                    global_to_local_map = {}
                    for entity_id, entity in enumerate(entities):
                        if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                        ):
                            entity["start"] = entity["start"] - index + add_num
                            entity["end"] = entity["end"] - index + add_num
                            global_to_local_map[entity_id] = len(entities_in_this_span)
                            entities_in_this_span.append(entity)
                    relations_in_this_span = []
                    for relation in relations:
                        if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                        ):
                            relations_in_this_span.append(
                                {
                                    "head": global_to_local_map[relation["head"]],
                                    "tail": global_to_local_map[relation["tail"]],
                                    "start_index": relation["start_index"] - index + add_num,
                                    "end_index": relation["end_index"] - index + add_num,
                                }
                            )
                    item.update(
                        {
                            "id": f"{doc['id']}_{chunk_id}",
                            "image": image,
                            "entities": entities_in_this_span,
                            "relations": relations_in_this_span,
                        }
                    )
                    yield f"{doc['id']}_{chunk_id}", item
