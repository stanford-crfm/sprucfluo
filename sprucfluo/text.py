"""Routines for dealing with text data, mostly for language modeling"""
from functools import partial
from typing import Optional, Iterator

from torch.utils.data import IterDataPipe
from transformers import BatchEncoding, PreTrainedTokenizerBase
from itertools import chain


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