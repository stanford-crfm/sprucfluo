##########
# THIS CODE IS STILL EXPERIMENTAL AND YOU SHOULDN'T USE IT YET.
########

# IterDataPipe for preprocessing data, tokenizing, and caching to disk
# The general file format we're going with is an apache parquet file with columns for the output of the tokenizer,
# A row is a single doc
# (We might add back in file metadata later)
# We don't want to have one giant file, so we'll split it up into chunks.
import functools
import json
import multiprocessing as mp
import os
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterDataPipe
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm
from transformers import BatchEncoding, AutoTokenizer, PreTrainedTokenizerFast

# As a heuristic, we're aiming for files that are around ~250MB
# Typically we're training on sequences of length ~1024 and batch size up to 512, so better to make it divisible by that.
# 4bytes * 512 * 1024 = 2Mi, so we'll go with 128 * 512 * 1024 = 67108864 tokens, which is about 256MiB

NUM_TOKENS_PER_FILE = 67108864


# TASKS:
# TODO: only do the caching on local_rank=0 so we do it once per device
# TODO: figure out how to best do multiple nodes
# TODO: figure out resuming preprocessing jobs
# TODO: make sure we handle reentrancy correctly in the dataset
# TODO: some kind of checksumming for ledger
# TODO: want to also handle being interrupted mid-file, and continuing where we left off.

LEDGER_FILE = "ledger.json"

def cache_encodings(
        doc_iter: Iterator[BatchEncoding],
        out_dir: str,
        file_template: str = 'docs-{}.parquet',
        num_tokens_per_file: int = NUM_TOKENS_PER_FILE) -> Iterator:
    """
    Given an iterator of documents, groups and caches them to disk, and yield them in batches of NUM_TOKENS_PER_FILE
    """
    file_index = 0
    current_writer: Optional[pq.ParquetWriter] = None
    current_num_tokens = 0
    tq: Optional[tqdm] = tqdm(desc=f"file {file_index} progress", total=num_tokens_per_file, unit="token")
    file_out: Optional[str] = None

    # list of (file_name, num_tokens), to be output at the end if we finish the whole iterator
    ledger = []

    def reset_writer(schema):
        nonlocal current_writer, tq, file_out
        file_out = f"{out_dir}/{file_template.format(file_index)}"
        os.makedirs(os.path.dirname(file_out), exist_ok=True)
        current_writer = pq.ParquetWriter(file_out, schema, version="2.6", compression="ZSTD")

        tq.reset()
        tq.set_description(f"file {file_index}")

    def as_record_batch(doc):
        names, columns = zip(*[(k, pa.array(v)) for k, v in doc.items() ])
        return pa.RecordBatch.from_arrays(list(columns), names)

    try:
        for doc in doc_iter:
            batch = as_record_batch(doc)
            batch_len = sum(len(t) for t in doc["input_ids"])
            if not current_writer:
                reset_writer(batch.schema)
            # NB: the elif means we'll write to this file if it's brand new even if the batch is too big
            # TODO: should we maybe split the batch if it's too big?
            elif current_num_tokens + batch_len > num_tokens_per_file:
                current_writer.close()
                current_writer = None
                ledger.append({"file_name": file_out, "num_tokens": current_num_tokens})
                yield file_out
                file_index += 1
                current_num_tokens = 0
                reset_writer(batch.schema)

            current_writer.write_batch(batch)
            current_num_tokens += batch_len
            tq.update(batch_len)

        if current_writer:
            tq.reset(current_num_tokens)
            tq.update(current_num_tokens)
            current_writer.close()
            if current_num_tokens > 0:
                ledger.append({"file_name": file_out, "num_tokens": current_num_tokens})
        # if we successfully wrote the whole iterator, we can write the ledger
        # TODO: maybe write the ledger incrementally, so we can resume where we left off if there's a crash
        # have to implement seeking in the inputs, so we can resume where we left off if there's a crash
        # alternatively, we can do one cache dir per input file, and have a separate ledger file for each input file
        with open(f"{out_dir}/{LEDGER_FILE}", "w") as f:
            json.dump(ledger, f)
        yield file_out
    except KeyboardInterrupt:
        current_writer.close()
        current_writer = None
        if file_out:
            os.remove(file_out)
        raise


def read_cache(file, flatten: bool=False) -> Iterator[BatchEncoding]:
    """ Reads the cache files produced by cache_and_group and yields tokenized sequences.
    If flatten is false, this returns the docs as they were presented to the caching process. If flatten is True,
    then the documents returned are actually concatenated documents, where the number is the number of documents
    presented as a batch to the caching process."""
    # for b in ArrowReader.read_table(file).to_batches():
    for b in pq.read_table(file).to_batches():
        if flatten:
            # insert a newaxis to the beginning so that it appears to be bs=1
            yield BatchEncoding(
                {b.field(i).name: b.column(i).values.to_numpy(zero_copy_only=True)[np.newaxis, :] for i in range(b.num_columns)}
            )
        else:
            yield BatchEncoding({b.field(i).name: b.column(i).to_numpy(zero_copy_only=False) for i in range(b.num_columns)})


class _FinishedSentinel:
    pass


class EagerCachingDataset(IterDataPipe[BatchEncoding]):
    def __init__(self,
                 doc_iter: IterDataPipe[BatchEncoding],
                 cache_dir: str,
                 num_tokens_per_file: int = NUM_TOKENS_PER_FILE,
                 file_template: str = 'docs-{}.parquet'):
        self.doc_iter = doc_iter
        self.cache_dir = cache_dir
        self.num_tokens_per_file = num_tokens_per_file
        self.file_template = file_template

        self.finished_file_queue = mp.Queue()
        self.caching_process: Optional[mp.Process] = None

    def fork_process(self):
        self.caching_process = mp.Process(
            target=self._fork_cache_process, daemon=False, name="PreCachingProcess")
        self.caching_process.start()

    def _fork_cache_process(self):
        for file in cache_encodings(self.doc_iter, self.cache_dir, self.file_template, self.num_tokens_per_file):
            self.finished_file_queue.put(file)

        self.finished_file_queue.put(_FinishedSentinel)

    def __iter__(self):
        # if we have a ledger file, read it and return the files
        if os.path.exists(f"{self.cache_dir}/{LEDGER_FILE}"):
            with open(f"{self.cache_dir}/{LEDGER_FILE}") as f:
                ledger = json.load(f)
            files = [f["file_name"] for f in ledger]
            for file in files:
                yield from read_cache(file)
            return
        # otherwise, start the caching process
        else:
            if self.caching_process is None:
                self.fork_process()

            while True:
                file = self.finished_file_queue.get()
                if file is _FinishedSentinel:
                    break
                yield from read_cache(file)


if __name__ == '__main__':
    from sprucfluo import *
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained('gpt2')

    data = ["https://storage.googleapis.com/pubmed-mosaic/wiki_en/train-00.jsonl.zst"]

    data = IterableWrapper(data)
    pipeline = data.open_file_by_fsspec_fancy(mode="r", compression="infer") \
        .readlines(return_path=False) \
        .map(json.loads) \
        .map(lambda x: x["text"]) \
        .then(batch_tokenize, tokenizer=tokenizer, return_attention_mask=False)
    ds = EagerCachingDataset(pipeline, "cache").flatmap(functools.partial(concatenate_and_group_texts, seq_len=512))
    total_tokens = 0
    for i, batch in enumerate(ds):
        if i < 10:
            print(batch)
        # if i > 10:
        #     break
    # for b in read_cache(['examples/foo/docs-0.arrow'], flatten=True):
    #     for grouped in concatenate_and_group_texts(b, seq_len=1024):
    #         print("ZZZ")
    #         print(grouped)
    #         print(tokenizer.decode(grouped['input_ids']))
    #     # print("<<<")
    #     # print(b.input_ids.shape)
    #     # print(len(b.input_ids))
    #     # print(tokenizer.decode(b['input_ids']))

