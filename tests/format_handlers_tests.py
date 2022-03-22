import io
import unittest

from format_handlers import *


class FormatHandlersTest(unittest.TestCase):
    def test_jsonl_samples_from_stream(self):
        test_data = [{"foo": "bar"}, {"foo": ["baz"]}, {"foo": {"qux": "quux"}}]
        as_jsonl = "\n".join([json.dumps(x) for x in test_data])

        url = "https://example.com/foo.jsonl"

        output = []
        for i, d in enumerate(test_data):
            d = d.copy()
            d["__index__"] = i
            d["__url__"] = url
            d["__key__"] = f"{url}:{i}"
            output.append(d)

        with io.StringIO(as_jsonl) as stream:
            samples = list(jsonl_samples_from_streams([{"url": url, "stream": stream}]))
            self.assertEqual(samples, output)






if __name__ == '__main__':
    unittest.main()
