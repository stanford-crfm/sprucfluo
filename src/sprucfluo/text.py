# Copyright 2022 The Board of Trustees of the Leland Stanford Junior University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Routines for dealing with text data, mostly for language modeling"""
import codecs
import json
import os.path
import re
import tarfile
from functools import partial
from typing import Optional, Iterator

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper
from transformers import BatchEncoding, PreTrainedTokenizerBase
from itertools import chain

try:
    import magic
except ImportError:
    magic = None


def concatenate_and_group_texts(encoding: BatchEncoding, seq_len: int, stride: Optional[int] = None) -> Iterator[BatchEncoding]:
    """Groups texts in a batch together. Typically, you'll want to use this with a fairly large
    set of texts, e.g. 1000 docs.

    Args:
        encoding: The batch of texts to concatenate and group.
        seq_len: The max length of sequences to emit
        stride: The stride to use when grouping texts. If None, then the stride is set to seq_len.

    Returns:
        An iterator of tokenized texts, one at a time.
    """
    concatenated = BatchEncoding(data={k: list(chain(*v)) for k, v in encoding.items()})
    total_length = len(concatenated.input_ids)
    stride = stride or seq_len

    # Drop the "very last" bit of the dataset that doesn't fit into block size...
    total_length = ((total_length - seq_len + stride) // stride) * stride

    # Split by Chunks of Maximum Length
    for i in range(0, total_length, stride):
        yield BatchEncoding(data={k: v[i:i + seq_len] for k, v in concatenated.data.items()})


# TODO: support truncation and padding
# TODO: support mlm
def tokenize_and_group_texts(pipe: IterDataPipe[str],
                             tokenizer: PreTrainedTokenizerBase,
                             seq_len: int,
                             batch_size: int = 1000,
                             stride: Optional[int] = None) -> IterDataPipe[BatchEncoding]:
    """Processes a set of texts for language modeling. Tokenizes, groups texts together, and splits them into sequences
    of length seq_len tokens each.

    Args:
        pipe: The pipe to process.
        tokenizer: The tokenizer to use.
        seq_len: The length of sequences to emit.
        batch_size: The batch size to use for tokenizing and grouping
        stride: The stride to use when grouping texts. If None, then the stride is set to seq_len.
    """
    return pipe.batch(batch_size=batch_size, wrapper_class=list)\
        .map(tokenizer)\
        .flatmap(partial(concatenate_and_group_texts, seq_len=seq_len, stride=stride))


def read_lm_text_file(file_path: str, stream: StreamWrapper, json_text_key: str = "text") -> Iterator[str]:
    rest_path, file_type = os.path.splitext(file_path)
    file_type = file_type.lstrip('.')

    if len(file_type) == 0 and stream.seekable() and magic is not None:
        file_type = sniff_file_type(stream.read(1024))
        stream.seek(0)

    # I hate this, but the OpenWebText format is a tar.xz file with a bunch of tar.xz files inside, each of
    # which contains a bunch of text files. We special case this to read the text files directly.
    # This matches https://github.com/leogao2/lm_dataformat
    if any(re.finditer(r'urlsf_subset', file_path)):
        yield from read_owt_subset(stream)
        return

    if file_type in file_handlers:
        yield from file_handlers[file_type](stream, json_text_key)
    else:
        msg = f"Unsupported file type: {file_type} for file {file_path}"
        if magic is None:
            msg += " (python-magic is not installed, so we can't sniff the file type from its contents)"
        raise ValueError(msg)


def sniff_file_type(data: bytes) -> Optional[str]:
    """Sniffs the file type of the given data.

    Args:
        data: The data to sniff.

    Returns:
        The file type.
    """
    mime = magic.from_buffer(data, mime=True)
    if mime == "application/json":
        return "jsonl"
    elif mime == "text/plain":
        return "txt"
    else:
        return mime

    # return None

# copied from lm_dataformat which is distributed under an MIT License:
# https://github.com/leogao2/lm_dataformat/blob/master/LICENSE
def read_owt_subset(stream):
    utf8reader = codecs.getreader('utf-8')
    tar = tarfile.TarFile(fileobj=stream)
    for name in tar.getmembers():
        fp = utf8reader(tar.extractfile(name))
        contents = fp.read()
        yield contents


def read_jsonl(stream: StreamWrapper, json_text_key: str = "text") -> Iterator[dict]:
    for line in stream:
        yield json.loads(line)[json_text_key]


def read_text(stream: StreamWrapper, json_text_key: str = "text") -> Iterator[dict]:
    yield stream.read()


file_handlers = {
    'text': read_text,
    'txt': read_text,
    'jsonl': read_jsonl,
}
