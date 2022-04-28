from typing import Union, List, Optional, Dict, Any

from torch.utils.data import IterDataPipe

from .files import expand_paths
from .text import read_lm_text_file

def load_corpus(paths: Union[str, List[str]],
                shard_by_rank: bool = True,
                cycle: bool = False,
                json_text_key: str = "text",
                extra_fsspec_args: Optional[Dict[str, Any]] = None,
                expand_globs: bool = False) -> IterDataPipe[str]:
    """
    Loads a corpus from a list of paths. Each element of the iterator will be the text from a single "document".

    Args:
        paths: A list of paths to the corpus. Will be expanded via braceexpand.
        shard_by_rank: If True, each shard will be assigned to a different rank, as per pytorch RANK
        cycle: If True, the corpus will be cycled to produce an infinite stream.
        json_text_key: The key in the JSON file to use as the text. Defaults to "text".
        extra_fsspec_args: Extra arguments to pass to fsspec. This can be used for authentication, etc.
        expand_globs: If True, will expand globs in the paths. This happens after the paths are expanded via braceexpand.
    """
    if extra_fsspec_args is None:
        extra_fsspec_args = {}
    paths = expand_paths(paths)

    if cycle:
        paths = paths.cycle()

    if shard_by_rank:
        paths = paths.shard_by_rank()

    return paths \
        .open_file_by_fsspec_fancy(expand_globs=expand_globs, mode="rb", compression="infer", **extra_fsspec_args) \
        .flatmap(lambda name_stream: read_lm_text_file(name_stream[0], name_stream[1], json_text_key))


__all__ = ["load_corpus"]