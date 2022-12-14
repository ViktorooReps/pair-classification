import json
from typing import Tuple, List, Callable

import torch
from torch import LongTensor
from torch.utils.data import Dataset
from transformers import BatchEncoding


class PairDataset(Dataset[Tuple[LongTensor, int]]):

    def __init__(self, examples: LongTensor, clusters: List[int], negative_samples: float = 1.0):
        self._examples = examples
        self._clusters = clusters

        total_examples = len(clusters)

        clusters = torch.tensor(clusters)
        same_cluster = (clusters.view(1, total_examples) == clusters.view(total_examples, 1)).view(-1)
        chosen_idx = torch.clone(same_cluster)

        # add some negative examples
        num_negative = int(same_cluster.sum() * negative_samples)
        negative_idxes = torch.randperm(len(same_cluster))[:num_negative]

        chosen_idx[negative_idxes] = True

        self._example_idxes = torch.concat(
            [
                torch.arange(total_examples).view(1, total_examples).repeat(total_examples, 1),
                torch.arange(total_examples).view(total_examples, 1).repeat(1, total_examples)
            ],
            dim=-1
        ).view(-1, 2)[chosen_idx]
        self._same_cluster = same_cluster[chosen_idx]

    def __getitem__(self, index) -> Tuple[LongTensor, int]:
        return self._examples[self._example_idxes[index]], int(self._same_cluster[index].item())

    def __len__(self) -> int:
        return len(self._example_idxes)


def get_train_datasets(tokenizer: Callable[[List[str]], BatchEncoding], *, split: float) -> Tuple[PairDataset, PairDataset]:
    with open('data/dev-dataset-task2022-04_preprocessed.json') as f:
        json_dataset = json.load(f)

    texts, labels = zip(*json_dataset)
    input_ids = tokenizer(texts)['input_ids']

    total_elements = len(json_dataset)
    train_elements = int(total_elements * split)

    train_data, dev_data = input_ids[:train_elements], input_ids[train_elements:]
    train_labels, dev_labels = labels[:train_elements], labels[train_elements:]

    return PairDataset(train_data, list(map(int, train_labels))), PairDataset(dev_data, list(map(int, dev_labels)))


def get_test_iterator() -> List[List[str]]:
    with open('data/dev-dataset-task2022-04_preprocessed.json') as f:
        json_dataset = json.load(f)
    return json_dataset
