import io
import unittest

from text import *


class MockTokenizer(object):
    def __init__(self, append_special_tokens=False):
        self.append_special_tokens = append_special_tokens

    def __call__(self, text):
        text = text.split()
        if self.append_special_tokens:
            text = text + ['[EOS]']
        return {"input_ids": text, "attention_mask": [1] * len(text)}


class TokenizeTests(unittest.TestCase):
    def test_tokenize_and_group_texts_short_texts(self):
        tokenizer = MockTokenizer(append_special_tokens=False)
        test_data = [
            {"text": "a a a a a a a a"}, # 8 tokens
            {"text": "b b b b b b b"}, # 7 tokens
            {"text": "q q q q q q q q"} # 8 tokens
        ]

        proc = tokenize_and_group_texts(tokenizer, max_seq_len=10)

        samples = list(proc(test_data))
        expected_input_ids = [
            ['a'] * 8 + ['b'] * 2,
            ['b'] * 5 + ['q'] * 5,
            ['q'] * 3
        ]

        expected_attention_mask = [
            [1] * 10,
            [1] * 10,
            [1] * 3
        ]

        self.assertEqual(len(samples), 3)
        for i in range(3):
            self.assertEqual(samples[i]['input_ids'], expected_input_ids[i])
            self.assertEqual(samples[i]['attention_mask'], expected_attention_mask[i])

    def test_tokenize_and_group_texts_long_texts(self):
        tokenizer = MockTokenizer(append_special_tokens=False)
        test_data = [
            {"text": "a a a a a a a a"}, # 8 tokens
            {"text": "b b b b b b b"}, # 7 tokens
            {"text": "q q q q q q q q"} # 8 tokens
        ]

        proc = tokenize_and_group_texts(tokenizer, max_seq_len=3)

        samples = list(proc(test_data))
        expected_input_ids = [
            ['a'] * 3,
            ['a'] * 3,
            ['a'] * 2 + ['b'] * 1,
            ['b'] * 3,
            ['b'] * 3,
            ['b'] * 1 + ['q'] * 2,
            ['q'] * 3,
            ['q'] * 3,
        ]

        expected_attention_mask = [[1] * len(s) for s in expected_input_ids ]

        self.assertEqual(len(samples), len(expected_input_ids))
        for i in range(3):
            self.assertEqual(samples[i]['input_ids'], expected_input_ids[i])
            self.assertEqual(samples[i]['attention_mask'], expected_attention_mask[i])








if __name__ == '__main__':
    unittest.main()
