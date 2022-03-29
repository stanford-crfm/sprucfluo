"""Exposes HF datasets as DataPipes"""
from typing import Union

import datasets
from torch.utils.data import IterDataPipe

AnyHfDataset = Union[datasets.Dataset, datasets.IterableDataset]
AnyHfDatasetDict = Union[datasets.DatasetDict, datasets.IterableDatasetDict]


class HFDatasetIterPipe(IterDataPipe):
    """A DataPipe that wraps a HF dataset and iterates over it."""

    def __init__(self, dataset: AnyHfDataset):
        """
        Args:
            dataset (HFDataset): HF dataset to wrap
        """
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)


