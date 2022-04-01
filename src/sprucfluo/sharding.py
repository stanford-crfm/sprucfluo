import itertools
from typing import TypeVar, Iterator, Sized

from torch.utils.data import functional_datapipe, IterDataPipe
from .utils import pytorch_worker_info

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe("shard_by_node")
class ShardByNodeDataPipe(IterDataPipe[T_co]):
    r"""
    A data pipe that shards the stream by node: each node will only see its fraction of the data. Typically used for
    sharding the data for distributed training. If you want workers, then use the built-in .sharding_filter()
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co]) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe

    def __iter__(self) -> Iterator[T_co]:
        rank, world_size, _, _ = pytorch_worker_info()
        if world_size == 1:
            return iter(self.source_datapipe)
        else:
            return itertools.islice(iter(self.source_datapipe), rank, None, world_size)

    def __len__(self):
        rank, world_size, _, _ = pytorch_worker_info()
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe) // world_size + \
                   (1 if (rank < len(self.source_datapipe) % world_size) else 0)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))