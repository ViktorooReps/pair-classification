from typing import List

from torch import Tensor, LongTensor
from transformers import AutoTokenizer, AutoModel, BatchEncoding, PreTrainedModel, PreTrainedTokenizer

from model.serializable import SerializableModel


class TextEncoder(SerializableModel):

    def __init__(self, bert_model: str):
        super(TextEncoder, self).__init__()
        self._bert_encoder: PreTrainedModel = AutoModel.from_pretrained(bert_model)
        self._bert_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(bert_model)

    @property
    def hidden_size(self) -> int:
        return self._bert_encoder.config.hidden_size

    def prepare_inputs(self, texts: List[str]) -> BatchEncoding:
        return self._bert_tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=self._bert_encoder.config.max_length,
            add_special_tokens=True
        )

    def forward(self, input_ids: LongTensor) -> Tensor:
        return self._bert_encoder(input_ids).last_hidden_state[:, 0]  # select [CLS] token representation
