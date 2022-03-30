import json

from torch.utils.data import IterDataPipe
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe
from transformers import AutoTokenizer, BatchEncoding

import sprucfluo
from sprucfluo import tokenize_and_group_texts
from torchdata.datapipes.iter import IterableWrapper

_HOST_URL = "https://mystic.the-eye.eu"
pile_data = [f"{_HOST_URL}/public/AI/pile/train/{i:0>2}.jsonl.zst" for i in range(30)]
pile_data = IterableWrapper(pile_data)

pubmed_data = [f"simplecache::https://storage.googleapis.com/pubmed-mosaic/pubmed-sharded/pubmedAbs_train.{i + 1}-of-128.jsonl.gz" for i in range(128)]
pubmed_data = IterableWrapper(pubmed_data)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def process_jsonl_corpus(urls: IterDataPipe[str]) -> IterDataPipe[BatchEncoding]:
    return urls.open_file_by_fsspec_fancy(mode="r", compression="infer") \
        .readlines(return_path=False) \
        .map(json.loads) \
        .map(lambda x: x["text"])\
        .then(tokenize_and_group_texts, tokenizer=tokenizer, seq_len=1024)



pubmed_data = process_jsonl_corpus(pubmed_data)
pile_data = process_jsonl_corpus(pile_data)

data = SampleMultiplexerDataPipe({pile_data: 5, pubmed_data: 5}, seed=0)

for encoded in data:
    print(encoded)
