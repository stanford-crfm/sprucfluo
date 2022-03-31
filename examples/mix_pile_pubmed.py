import json

from torch.utils.data import IterDataPipe
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe
from transformers import AutoTokenizer, BatchEncoding

import sprucfluo
from sprucfluo import tokenize_and_group_texts

pile_data = sprucfluo.load_corpus("https://mystic.the-eye.eu/public/AI/pile/train/{00..29}.jsonl.zst")
pubmed_data = sprucfluo.load_corpus("gcs://pubmed-mosaic/pubmed-sharded/pubmedAbs_train.{1..128}-of-128.jsonl.gz")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

pubmed_data = pubmed_data.then(tokenize_and_group_texts, tokenizer=tokenizer, seq_len=1024)
pile_data = pile_data.then(tokenize_and_group_texts, tokenizer=tokenizer, seq_len=1024)

data = SampleMultiplexerDataPipe({pile_data: 5, pubmed_data: 5}, seed=0)

for encoded in data:
    print(encoded)
