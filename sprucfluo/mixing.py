# desired api:
# streams = {
#   "webtext": [doc streams]
#   "pubmed": [doc streams]
# }
# where doc streams are their own pipelines, producing docs

# weights = {
#   "webtext": 0.8,
#   "pubmed": 0.2,
# }
#
#
# Pipeline(sample(streams, weights), preprocessing steps)
from random import Random
from typing import Dict, Iterable, Optional, Union


def sample(streams: Dict[str, Iterable],
           weights: Optional[Dict[str, float]] = None,
           rng: Optional[Union[int, Random]] = None,
           stop_policy: str = "raise") -> Iterable:
    """
    Sample from streams according to weights.
    If weights is None, uniform sampling is used.

    Args:
        streams: dict of streams
        weights: dict of weights, or None, same keys as streams. If None, uniform sampling is used.
        rng: random number generator, or seed for Random()
        stop_policy: "stop", "skip", "loop", "raise"
            These have the following meanings:
            "stop": stop after the first stream is exhausted
            "skip": skip exhausted streams
            "loop": loop over streams, restarting them (via __iter__) when exhausted
            "raise": raise StopIteration when the first stream is exhausted
    """
    if weights is None:
        weights = {k: 1.0 / len(streams) for k in streams}
        total = 1.0
    else:
        assert streams.keys() == weights.keys()
        total = sum(weights.values())

    # TODO: think through when to use worker-specific rng
    if rng is None:
        rng = Random()
    elif isinstance(rng, int):
        rng = Random(rng)

    assert stop_policy in ["stop", "skip", "loop", "raise"]

    iterators = {k: iter(v) for k, v in streams.items()}
    while True:
        r = rng.random() * total
        for k, v in weights.items():
            if r < v:
                try:
                    yield next(iterators[k])
                except StopIteration:
                    if stop_policy == "raise":
                        raise RuntimeError(f"{k} stream exhausted")
                    elif stop_policy == "skip":
                        del iterators[k]
                        del weights[k]
                        total -= v
                        break
                    elif stop_policy == "loop":
                        iterators[k] = iter(streams[k])
                        yield next(iterators[k])
                    elif stop_policy == "stop":
                        return
                    else:
                        raise ValueError(f"Unknown stop_policy: {stop_policy}")
                break
            else:
                r -= v




