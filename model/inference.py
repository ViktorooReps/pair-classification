import logging
import time
from copy import deepcopy
from itertools import chain, starmap
from os import cpu_count, environ
from pathlib import Path
from typing import Dict, List, TypeVar, Optional, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor, LongTensor
from torch.nn import Module
from torch.onnx import export
from transformers import TensorType
from transformers.convert_graph_to_onnx import quantize
from transformers.onnx import FeaturesManager, OnnxConfig

from datamodel.metric import bcubed_f1
from model.pair_classifier import PairClassifier
from model.serializable import SerializableModel
from model.utils import to_numpy

torch.set_num_threads(cpu_count() // 2)

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count() // 2)
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

logger = logging.getLogger(__name__)

try:
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions
except:
    logger.warning('Could not import ONNX inference tools!')


def batch_cluster_inputs(
        example_representations: Iterable[Tensor],
        cluster_ids: Iterable[int],
        *,
        batch_size: int
) -> Tuple[Tensor, List[int]]:

    curr_batch: List[Tensor] = []
    curr_ids: List[int] = []

    for example, idx in zip(example_representations, cluster_ids):
        if len(curr_batch) == batch_size:
            yield torch.stack(curr_batch), curr_ids
            curr_batch = []
            curr_ids = []

        curr_batch.append(example)
        curr_ids.append(idx)

    if len(curr_batch):
        yield torch.stack(curr_batch), curr_ids


_Model = TypeVar('_Model', bound=Module)


class PairCluster(SerializableModel):

    def __init__(self, pair_classifier: PairClassifier, batch_size: int):
        super().__init__()
        self._classifier = pair_classifier
        self._classifier.eval()

        self._batch_size = batch_size
        self._clusters: List[List[Tensor]] = []

    def train(self: _Model, mode: bool = True) -> _Model:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, text: str) -> str:
        compare_encoding = self._classifier.prepare_inputs([text]).encodings[0]
        compare_representation = self._classifier.encode(compare_encoding.ids.unsqueeze(0))

        if not len(self._clusters):
            self._clusters.append([compare_representation])
            return '0'

        predictions = np.zeros(len(self._clusters), dtype=float)
        cluster_sizes = np.array(list(map(len, self._clusters)), dtype=float)

        cluster_ids = chain.from_iterable(starmap(lambda idx, size: [idx] * size, enumerate(cluster_sizes)))
        cluster_examples = chain.from_iterable(self._clusters)

        for representations, ids in batch_cluster_inputs(cluster_examples, cluster_ids, batch_size=self._batch_size):
            batch_size = len(representations)
            compare_representations = compare_representation.repeat(batch_size, 1)
            logits = self._classifier.head_forward(representations, compare_representations)

            predictions = torch.argmax(logits, dim=-1)
            for idx, prediction in zip(ids, predictions):
                predictions[idx] += prediction  # 1 for predicted, 0 otherwise

        probs = predictions / cluster_sizes
        best_cluster_id = np.argmax(probs)
        best_cluster_prob = probs[best_cluster_id]

        if best_cluster_prob < 0.5:
            # create new cluster
            best_cluster_id = len(self._clusters)
            self._clusters.append([compare_representation])
        else:
            # add to existing cluster
            self._clusters[best_cluster_id].append(compare_representation)

        return str(best_cluster_id)

    def optimize(
            self,
            onnx_dir: Path,
            quant: bool = True,
            fuse: bool = True,
            opset_version: int = 13,
            do_constant_folding: bool = True
    ) -> None:
        onnx_model_path = onnx_dir.joinpath('model.onnx')
        onnx_optimized_model_path = onnx_dir.joinpath('model-optimized.onnx')

        # load config
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self._classifier._encoder._bert_encoder)
        onnx_config: OnnxConfig = model_onnx_config(self._classifier._encoder._bert_encoder.config)

        model_inputs = onnx_config.generate_dummy_inputs(self._classifier._encoder._bert_tokenizer, framework=TensorType.PYTORCH)
        dynamic_axes = {0: 'batch', 1: 'sequence'}
        # export to onnx
        export(
            self._classifier._token_encoder,
            ({'input_ids': model_inputs['input_ids']},),
            f=onnx_model_path.as_posix(),
            verbose=False,
            input_names=('input_ids',),
            output_names=('last_hidden_state',),
            dynamic_axes={'input_ids': dynamic_axes, 'last_hidden_state': dynamic_axes},
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
        )

        if fuse:
            opt_options = FusionOptions('bert')
            opt_options.enable_embed_layer_norm = False

            optimizer.optimize_model(
                str(onnx_model_path),
                'bert',
                num_heads=12,
                hidden_size=768,
                optimization_options=opt_options
            ).save_model_to_file(str(onnx_optimized_model_path))

            onnx_model_path = onnx_optimized_model_path

        if quant:
            onnx_model_path = quantize(onnx_model_path)

        self._classifier._encoder._bert_encoder = ONNXOptimizedEncoder(onnx_model_path)


class ONNXOptimizedEncoder(Module):

    def __init__(self, onnx_path: Path):
        super().__init__()
        self._onnx_path = onnx_path
        self._session: Optional[InferenceSession] = None

    def __getstate__(self):
        state = deepcopy(self.__dict__)
        state.pop('_session')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._session = None

    def _start_session(self) -> None:
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        self._session = InferenceSession(self._onnx_path.as_posix(), options, providers=['CPUExecutionProvider'])
        self._session.disable_fallback()

    def forward(self, input_ids: LongTensor, **_) -> Dict[str, Tensor]:
        if self._session is None:
            logger.info(f'Starting inference session for {self._onnx_path}.')
            start_time = time.time()
            self._start_session()
            logger.info(f'Inference started in {time.time() - start_time:.4f}s.')

        # Run the model (None = get all the outputs)
        return {
            'last_hidden_state': torch.tensor(self._session.run(
                None,
                {
                    'input_ids': to_numpy(input_ids)
                }
            )[0])
        }


def evaluate(model: PairCluster, texts: Iterable[str], ground_truth: Iterable[str]) -> Dict[str, float]:
    predictions = list(map(model, texts))
    return {'bcubed_f1': bcubed_f1(predictions, ground_truth)}
