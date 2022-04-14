from typing import TypeVar, List, Union

from torch.utils.data import IterDataPipe

from .files import FancyFSSpecFileOpenerIterDataPipe
from .hf_dataset import HFDatasetIterPipe
from .sharding import ShardByNodeDataPipe
from .text import concatenate_and_group_texts, tokenize_and_group_texts, read_lm_text_file
from .corpus import load_corpus

_T = TypeVar("_T", contravariant=True)
_U = TypeVar("_U", covariant=True)





def _then_data_pipe(data_pipe: IterDataPipe[_T], fn, *args, **kwargs) -> IterDataPipe[_U]:
    """
    A helper function to apply a function to a data pipe. Syntax is:
    data_pipe.then(fn, *args, **kwargs)
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
    'read_lm_text_file',
    'tokenize_and_group_texts',
    'ShardByNodeDataPipe',
    'expand_paths'
]

init()
