from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional, Iterable, Dict, TypeVar

import torch
from torch import LongTensor, Tensor
from torch.nn import Sequential, LayerNorm, Dropout, Linear, ReLU, CrossEntropyLoss
from transformers import BatchEncoding

from model.encoder import TextEncoder
from model.serializable import SerializableModel


@dataclass
class ModelArguments:
    bert_model: str = field(metadata={'help': 'BERT encoder'})
    dropout: float = field(metadata={'help': 'Dropout rate'})
    save_path: str = field(metadata={'help': 'Path to saved model'})


_Model = TypeVar('_Model', bound='PairClassifier')


class PairClassifier(SerializableModel):

    def __init__(self, bert_model: str, dropout: float):
        super().__init__()
        self._encoder = TextEncoder(bert_model)

        # input_size = 2 * self._encoder.hidden_size
        # self._head = Sequential(
        #     LayerNorm(input_size),
        #     Dropout(dropout),
        #     Linear(input_size, input_size // 4),
        #     ReLU(),
        #
        #     LayerNorm(input_size // 4),
        #     Dropout(),
        #     Linear(input_size // 4, input_size // 16),
        #     ReLU(),
        #
        #     LayerNorm(input_size // 16),
        #     Dropout(),
        #     Linear(input_size // 16, 2),
        # )

        input_size = 2 * self._encoder.hidden_size
        self._head = Sequential(
            Dropout(dropout),
            Linear(input_size, input_size // 4),
            ReLU(),

            Dropout(),
            Linear(input_size // 4, input_size // 16),
            ReLU(),

            Dropout(),
            Linear(input_size // 16, 2),
        )

        # input_size = self._encoder.hidden_size
        # self._head = Sequential(
        #     Dropout(dropout),
        #     Linear(input_size, 2)
        # )

    @classmethod
    def from_args(cls: _Model, args: ModelArguments) -> _Model:
        return cls(bert_model=args.bert_model, dropout=args.dropout)

    def prepare_inputs(self, texts: List[str]) -> BatchEncoding:
        return self._encoder.prepare_inputs(texts)

    @staticmethod
    def collate_examples(examples: Iterable[Tuple[LongTensor, int]]) -> Dict[str, LongTensor]:
        pairs, labels = zip(*examples)
        return {
            'pairs': torch.stack(pairs).long(),
            'labels': torch.tensor(labels, dtype=torch.long).long()
        }

    def head_forward(self, base_representation: Tensor, compare_representation: Tensor) -> Tensor:
        concat_representations = torch.concat([base_representation, compare_representation], dim=-1)  # (BATCH, 2 * HIDDEN)
        # return self._head(base_representation - compare_representation)
        return self._head(concat_representations)

    def encode(self, tokenized_texts: LongTensor) -> Tensor:
        return self._encoder(tokenized_texts)

    def forward(self, pairs: LongTensor, labels: Optional[LongTensor] = None) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        :param pairs: input ids of shape (BATCH, 2, LENGTH)
        :param labels: ground truth (0/1) or shape (BATCH)
        :return: predicted logits and loss of labels is given
        """
        batch_size, _, length = pairs.shape
        representations = self.encode(pairs.view(-1, length)).view(batch_size, 2, -1)  # (BATCH, 2, HIDDEN)
        logits = self.head_forward(representations[:, 0], representations[:, 1])

        if labels is not None:
            loss = CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
