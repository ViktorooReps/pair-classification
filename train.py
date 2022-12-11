import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from transformers import Trainer, HfArgumentParser, TrainingArguments, EvalPrediction
from transformers.modeling_utils import unwrap_model

from datamodel.configuration import get_train_datasets, get_test_iterator
from model.inference import evaluate, PairCluster
from model.pair_classifier import PairClassifier, ModelArguments

logger = logging.getLogger(__name__)


def compute_metrics(evaluation_results: EvalPrediction) -> Dict[str, float]:
    y_true = evaluation_results.label_ids
    y_pred = np.argmax(evaluation_results.predictions, axis=-1)

    precision_scores = precision_score(y_true, y_pred, average='none')
    recall_scores = recall_score(y_true, y_pred, average='none')
    return {
        'precision_0': precision_scores[0],
        'precision_1': precision_scores[1],
        'recall_0': recall_scores[0],
        'recall_1': recall_scores[1],
        'accuracy': accuracy_score(y_true, y_pred)
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser(dataclass_types=[ModelArguments, TrainingArguments])
    model_args, training_args = parser.parse_args_into_dataclasses()

    training_args: TrainingArguments
    model_args: ModelArguments

    model: PairClassifier = PairClassifier.from_args(model_args)
    train_dataset, dev_dataset = get_train_datasets(tokenizer=model.prepare_inputs, split=0.9)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=model.collate_examples,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    def normalize(d: Dict[str, Any]) -> Dict[str, str]:
        return {k: str(v) for k, v in d.items()}

    trained_model: PairClassifier = unwrap_model(trainer.model_wrapped)
    trained_model.cpu()
    inference_model = PairCluster(trained_model, batch_size=training_args.eval_batch_size)
    inference_model.save(Path(model_args.save_path))

    inference_model.cuda()

    model_predictions = []
    ground_truth = []

    evaluate(inference_model, *zip(*get_test_iterator()))
