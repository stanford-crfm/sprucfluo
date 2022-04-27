import functools
import json

from transformers import AutoTokenizer

from sprucfluo import preprocess
from sprucfluo.text import tokenize_and_group_texts, batch_tokenize
from torchdata.datapipes.iter import IterableWrapper


_HOST_URL = "https://mystic.the-eye.eu"
data = [f"{_HOST_URL}/public/AI/pile/train/{i:0>2}.jsonl.zst" for i in range(30)]

data = IterableWrapper(data)

tokenizer = AutoTokenizer.from_pretrained("gpt2", return_attention_mask=False)


for text in preprocess.cache_encodings(data.open_file_by_fsspec_fancy(mode="r", compression="infer")\
            .readlines(return_path=False)\
            .map(json.loads)\
            .map(lambda x: x["text"])\
            .then(batch_tokenize, tokenizer=tokenizer), "foo"):
    print(text)
