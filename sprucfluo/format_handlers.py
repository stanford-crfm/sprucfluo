import bz2
import gzip
import json
import lzma
import os
from typing import Optional, Callable, Dict, Union

from webdataset import reraise_exception, filters
from webdataset.cache import cached_url_opener, pipe_cleaner
from webdataset.tariterators import url_opener

_KeyGeneratorFn = Optional[Callable[[Dict], str]]


def jsonl_samples_from_streams(streams, id_key: Optional[Union[_KeyGeneratorFn, str]] = None,
                               handler=reraise_exception):
    """
    This function generates an iterator of samples from an iterator over jsonl stream samples.
    The sources in the stream should have a url and a stream key.
    In addition to the fields in the json dict, this method adds __url__ and __index__ (line in the file)

    If provided, id_key is a function that takes a dict sample and returns a unique identifier for the sample.
    (The default is to use the __url__ and index of the file)

    Cribbed from tariterators.py tar_file_expander
    """
    for source in streams:
        try:
            url = source["url"]
            assert isinstance(source, dict)
            assert "stream" in source
            for i, sample in enumerate(_iterate_jsonl(source["stream"])):
                # this is gross but it's how tar_file_expander works
                assert "__url__" not in sample
                assert "__index__" not in sample
                sample["__url__"] = url
                sample["__index__"] = i

                # webdatasets likes this __key__ field
                if isinstance(id_key, str):
                    sample["__key__"] = sample[id_key]
                elif id_key is not None:
                    sample["__key__"] = id_key(sample)
                else:
                    sample["__key__"] = f"{url}:{i}"

                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


def jsonl_samples(src,
                  id_key: Optional[Union[_KeyGeneratorFn, str]] = None,
                  handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    samples = jsonl_samples_from_streams(streams, id_key=id_key, handler=handler)
    return samples


# TODO: this is how webdatasets does it, but it's gross! much better to use a pipeline
def cached_jsonl_samples(src,
                         id_key: Optional[Union[_KeyGeneratorFn, str]] = None,
                         handler=reraise_exception,
                         cache_size=-1,
                         cache_dir=None,
                         url_to_name=pipe_cleaner,
                         verbose=False,
                         always=False,
                         ):
    streams = cached_url_opener(src, handler=handler, cache_size=cache_size, cache_dir=cache_dir,
                                url_to_name=url_to_name, verbose=verbose, always=always)
    samples = jsonl_samples_from_streams(streams, id_key=id_key, handler=handler)
    return samples


jsonl_to_samples = filters.pipelinefilter(jsonl_samples)
cache_jsonl_to_samples = filters.pipelinefilter(cached_jsonl_samples)
jsonl_streams_to_samples = filters.pipelinefilter(jsonl_samples_from_streams)


def _iterate_jsonl(stream):
    for line in stream:
        yield json.loads(line)


def zstd_decompress(stream):
    import zstandard
    zstandard.open(stream)


def _add_handlers(d, keys, value):
    if isinstance(keys, str):
        keys = keys.split()
    for k in keys:
        d[k] = value


AUTO_DECOMPRESSORS = {}
_add_handlers(AUTO_DECOMPRESSORS, "zstd zst", zstd_decompress)
_add_handlers(AUTO_DECOMPRESSORS, "gz zip", gzip.open)
_add_handlers(AUTO_DECOMPRESSORS, "bz2", bz2.open)
_add_handlers(AUTO_DECOMPRESSORS, "xz xzip", lzma.open)


def register_decompressor(ext, func):
    AUTO_DECOMPRESSORS[ext] = func


def autodecompress_stream(sample, stream_key: str = "stream", url_key: str = "url"):
    url = sample[url_key]
    ext = os.path.splitext(url)[1]
    decompressor = AUTO_DECOMPRESSORS.get(ext)
    if decompressor is not None:
        new_sample = sample.copy()
        new_sample[stream_key] = decompressor(sample[stream_key])
        new_sample[url_key] = url[:-len(ext)]
        return new_sample
    else:
        return sample


zstd_decompress_streams = filters.map_dict(stream=zstd_decompress)
gzip_decompress_streams = filters.map_dict(stream=gzip.open)
bz2_decompress_streams = filters.map_dict(stream=bz2.open)
xz_decompress_streams = filters.map_dict(stream=lzma.open)

autodecompress = filters.map(autodecompress_stream)
