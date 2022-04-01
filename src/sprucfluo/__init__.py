from typing import TypeVar, List, Union, Optional, Any

from braceexpand import braceexpand
from torch.utils.data import IterDataPipe
from torchdata.datapipes.iter import IterableWrapper

from .fancy_files import FancyFSSpecFileOpenerIterDataPipe
from .hf_dataset import HFDatasetIterPipe
from .sharding import ShardByNodeDataPipe
from .text import concatenate_and_group_texts, tokenize_and_group_texts, read_lm_text_file

_T = TypeVar("_T", contravariant=True)
_U = TypeVar("_U", covariant=True)


def expand_paths(paths: Union[str, List[str]]) -> IterDataPipe[str]:
    """
    Expand a list of URLs into a data pipe of URLs.
    """
    if isinstance(paths, str):
        paths = [paths]

    # TODO: we could be fancier and support globbing, but it makes things a bit more complicated, but only a bit
    return IterableWrapper([p for path in paths for p in braceexpand(path)])


def load_corpus(paths: Union[str, List[str]],
                shard_by_node: bool = True,
                cycle: bool = False,
                json_text_key: str = "text",
                extra_fsspec_args: Optional[dict[str, Any]] = None) -> IterDataPipe[str]:
    if extra_fsspec_args is None:
        extra_fsspec_args = {}
    paths = expand_paths(paths)

    if cycle:
        paths = paths.cycle()

    if shard_by_node:
        paths = paths.shard_by_node()

    return paths \
        .open_file_by_fsspec_fancy(mode="r", compression="infer", **extra_fsspec_args) \
        .flatmap(lambda name_stream: read_lm_text_file(name_stream[0], name_stream[1], json_text_key))


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
]

init()
