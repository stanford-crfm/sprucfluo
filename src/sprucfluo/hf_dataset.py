"""Exposes HF datasets as DataPipes"""
from typing import Union, Any

try:
    import datasets
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

from torch.utils.data import IterDataPipe

if _HAS_DATASETS:
    AnyHfDataset = Union[datasets.Dataset, datasets.IterableDataset]
    AnyHfDatasetDict = Union[datasets.DatasetDict, datasets.IterableDatasetDict]
else:
    AnyHfDataset = AnyHfDatasetDict = Any


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


