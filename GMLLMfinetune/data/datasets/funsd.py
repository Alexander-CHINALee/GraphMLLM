# coding=utf-8

import json
import os

import datasets
from .XYCut import augment_xy_cut
from GMLLMfinetune.data.utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "position_ids": datasets.Sequence(datasets.Value("int64")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "line_index": datasets.Sequence(datasets.Value("int64")),
                    "line_bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            line_index = []
            line_bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            ordered_forms, item_bboxes = [],[]
            for item in data["form"]:
                item_bboxes.append(item['box'])
            for item in data["form"]:
                words, label, line_bbox = item["words"], item["label"], item["box"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                line_bboxes.append(normalize_bbox(line_bbox, size))
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        bboxes.append(normalize_bbox(w["box"], size))
                        line_index.append(len(line_bboxes)-1)
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    bboxes.append(normalize_bbox(words[0]["box"], size))
                    line_index.append(len(line_bboxes)-1)
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        bboxes.append(normalize_bbox(w["box"], size))
                        line_index.append(len(line_bboxes)-1)
                
            # reorder
            line_word_group = [[] for _ in range(len(line_bboxes))]
            for word_i, line_i in enumerate(line_index):
                line_word_group[line_i].append(word_i)
            orders = augment_xy_cut(line_bboxes,direction='y')
            wpos_ids = []
            for line_i in orders:
                wpos_ids += line_word_group[line_i]
            
            position_ids = [-1 for _ in range(len(wpos_ids))]
            pos_i = 0
            for w_i in wpos_ids:
                position_ids[w_i] = pos_i
                pos_i += 1
            assert len(position_ids) == len(bboxes) and -1 not in position_ids
            
            yield guid, {"id": str(guid), "tokens": tokens, "position_ids": position_ids, "bboxes": bboxes, "line_bboxes": line_bboxes,\
                         "line_index": line_index, "ner_tags": ner_tags, "image": image}
