"""Routines for dealing with text data, mostly for language modeling"""
from typing import Dict, Iterator, Any

import webdataset
from transformers import PreTrainedTokenizerBase


def default_doc_slice(doc, text_key, tokenizer_outputs, start_token, end_token):
    """Creates a document that has just the tokens from start_token to end_token from the original doc"""
    doc = doc.copy()
    for k, v in tokenizer_outputs.items():
        doc[k] = v[start_token:end_token]

    if start_token != 0 and end_token != len(tokenizer_outputs["input_ids"]):
        if "offset_mapping" in tokenizer_outputs:
            offset_mapping = tokenizer_outputs["offset_mapping"]
            start_char = offset_mapping[start_token][0]
            end_char = offset_mapping[end_token][1]
            doc[text_key] = doc[text_key][start_char:end_char]
            doc["__char_offsets__"] = [(start_char, end_char)]

        if "__key__" in doc:
            doc["__key__"] = f"{doc['__key__']}:{start_token}-{end_token}"

    doc["__token_offsets__"] = [(start_token, end_token)]

    return doc


def default_doc_merge(doc1, doc2):
    """Merges two documents by concatenating everything. the __key__ is concatenate with a `::` separator"""
    doc = doc1.copy()
    for k, v in doc2.items():
        if k == "__key__":
            doc[k] = f"{doc['__key__']}::{v}"
        elif isinstance(v, str):
            doc[k] = f"{doc[k]}::{v}"
        elif k in doc:
            doc[k] += v
        else:
            doc[k] = v
    return doc


def _tokenize_and_group_texts(src,
                              tokenizer: PreTrainedTokenizerBase,
                              max_seq_len: int,
                              merge_docs=default_doc_merge,
                              slice_doc=default_doc_slice,
                              text_key: str = "text") -> Iterator[Dict[str, Any]]:
    # handles overflowing docs and combining keys
    buffered_len = 0
    buffered_doc = {}

    # TODO: consider adding support for pre-batched documents
    for doc in src:
        tokenized = tokenizer(doc[text_key])
        doc = doc.copy()

        num_tokens_in_this_doc = len(tokenized["input_ids"])

        if buffered_len > 0:
            offset = min(num_tokens_in_this_doc, max_seq_len - buffered_len)
            sliced = slice_doc(doc, text_key, tokenized, 0, offset)

            buffered_doc = merge_docs(buffered_doc, sliced)
            buffered_len += offset
            assert buffered_len <= max_seq_len
            if buffered_len == max_seq_len:
                yield buffered_doc
                buffered_len = 0
                buffered_doc = {}
            else:
                # doc isn't full yet, keep buffering
                continue
        else:
            offset = 0

        assert buffered_len == 0

        while offset + max_seq_len < num_tokens_in_this_doc:
            sliced = slice_doc(doc, text_key, tokenized, offset, offset + max_seq_len)
            offset += max_seq_len
            yield sliced

        if offset != num_tokens_in_this_doc:
            buffered_doc = slice_doc(doc, text_key, tokenized, offset, num_tokens_in_this_doc)
            buffered_len = len(buffered_doc["input_ids"])

    yield buffered_doc


tokenize_and_group_texts = webdataset.filters.pipelinefilter(_tokenize_and_group_texts)
