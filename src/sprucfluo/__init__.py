from typing import Callable, TypeVar

from torch.utils.data import IterDataPipe

from sprucfluo.fancy_files import FancyFSSpecFileOpenerIterDataPipe
from sprucfluo.hf_dataset import HFDatasetIterPipe
from sprucfluo.text import concatenate_and_group_texts, tokenize_and_group_texts
from sprucfluo.sharding import ShardByNodeDataPipe, ShardByWorkerDataPipe

_T = TypeVar("_T", contravariant=True)
_U = TypeVar("_U", covariant=True)


def _then_data_pipe(data_pipe: IterDataPipe[_T], fn: Callable[[IterDataPipe[_T], ...], IterDataPipe[_U]], *args, **kwargs) -> IterDataPipe[_U]:
    """
    A helper function to apply a function to a data pipe.
    """
    return fn(data_pipe, *args, **kwargs)


def init():
    IterDataPipe.register_function("then", _then_data_pipe)
    """
    Initialize the sprucfluo package, which registers a bunch of methods and such with IterDataPipe
    """
    pass


__all__ = [
    'FancyFSSpecFileOpenerIterDataPipe',
    'HFDatasetIterPipe',
    'concatenate_and_group_texts',
    'tokenize_and_group_texts',
    'ShardByNodeDataPipe',
    'ShardByWorkerDataPipe',
]

init()