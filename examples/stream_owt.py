import json

from torch.utils.data import IterDataPipe
from torchdata.datapipes.iter.util.samplemultiplexer import SampleMultiplexerDataPipe
from transformers import AutoTokenizer, BatchEncoding

import sprucfluo
import sprucfluo.corpus
from sprucfluo import tokenize_and_group_texts

owt_data = sprucfluo.corpus.load_openwebtext()

tokenizer = AutoTokenizer.from_pretrained("gpt2")

pile_data = owt_data.then(tokenize_and_group_texts, tokenizer=tokenizer, seq_len=1024)

for encoded in pile_data:
    print(encoded)
