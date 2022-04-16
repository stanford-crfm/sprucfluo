from typing import Union, List, Optional, Dict, Any

from torch.utils.data import IterDataPipe

from .files import expand_paths
from .text import read_lm_text_file

# corpus urls:
# OWT: https://storage.googleapis.com/pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz
#      https://storage.googleapis.com/pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz
# medical: "https://storage.googleapis.com/pubmed-mosaic/plain-medical-text-sharded/plain_medical_text_train.{1..128}-of-128.jsonl.gz",
#          "https://storage.googleapis.com/pubmed-mosaic/plain-medical-text-sharded/plain_medical_text_val.{1..8}-of-8.jsonl.gz",


def load_corpus(paths: Union[str, List[str]],
                shard_by_node: bool = True,
                cycle: bool = False,
                json_text_key: str = "text",
                extra_fsspec_args: Optional[Dict[str, Any]] = None,
                expand_globs: bool = False) -> IterDataPipe[str]:
    if extra_fsspec_args is None:
        extra_fsspec_args = {}
    paths = expand_paths(paths)

    if cycle:
        paths = paths.cycle()

    if shard_by_node:
        paths = paths.shard_by_node()

    return paths \
        .open_file_by_fsspec_fancy(expand_globs=expand_globs, mode="rb", compression="infer", **extra_fsspec_args) \
        .flatmap(lambda name_stream: read_lm_text_file(name_stream[0], name_stream[1], json_text_key))


__all__ = ["load_corpus", "expand_paths"]