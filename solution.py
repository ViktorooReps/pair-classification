import logging
import time
from pathlib import Path
from typing import Iterable, Set, Tuple, Dict, Sequence

import numpy as np

from datamodel.configuration import get_descriptions, DatasetName
from datamodel.example import TypedSpan
from datamodel.reader.nerel import get_dataset_files, read_annotation, read_text
from datamodel.utils import DatasetType, invert
from model.inference import InferenceBinder, evaluate


class Solution:

    model_path = Path('onnx/main.pkl')
    exclude_filenames = {
        '165459_text', '176167_text', '178485_text', '192238_text',
        '193267_text', '193946_text', '194112_text', '2021',
        '202294_text', '2031', '209438_text', '209731_text', '546860_text'
    }

    @classmethod
    def predict(cls, texts: Sequence[str]) -> Iterable[Set[Tuple[TypedSpan]]]:
        model = InferenceBinder.load(cls.model_path)
        return model(list(texts))

    @classmethod
    def evaluate(cls):
        text_files, annotation_files = get_dataset_files(Path('data/nerel'), DatasetType.TEST, exclude_filenames=cls.exclude_filenames)
        test_categories = sorted(get_descriptions(DatasetName.NEREL).keys())

        ground_truth = list(map(read_annotation, annotation_files))
        texts = list(map(read_text, text_files))

        model = InferenceBinder.load(cls.model_path)
        model_predictions = model(texts)

        return evaluate(model_predictions, ground_truth, test_categories)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    Solution.evaluate()
    end_time = time.time()

    print(f'Test time: {end_time - start_time:.4f}s')
