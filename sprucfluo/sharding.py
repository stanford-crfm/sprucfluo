import itertools
from typing import TypeVar, Iterator

from torch.utils.data import functional_datapipe, IterDataPipe
from webdataset.utils import pytorch_worker_info

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe("shard_by_node")
class ShardByNodeDataPipe(IterDataPipe[T_co]):
    r"""
    A data pipe that shards the stream by node: each node will only see its fraction of the data. Typically used for
    sharding the data for distributed training.
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co]) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.rank, self.world_size, _, _ = pytorch_worker_info()

    def __iter__(self) -> Iterator[T_co]:
        if self.world_size == 1:
            return self.source_datapipe
        return itertools.islice(self.source_datapipe, self.rank, None, self.world_size)

    def __len__(self) -> int:
        all_len = len(self.source_datapipe)
        return all_len // self.world_size + (1 if all_len % self.world_size > self.rank else 0)


@functional_datapipe("shard_by_worker")
class ShardByWorkerDataPipe(IterDataPipe[T_co]):
    r"""
    A data pipe that shards the stream by worker: each worker will only see its fraction of the data. Typically used
    for data loaders within a node to ensure that each worker sees a different subset of the data.
    """

    def __init__(self, source_datapipe: IterDataPipe[T_co]) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        _, _, self.worker, self.num_workers = pytorch_worker_info()

    def __iter__(self) -> Iterator[T_co]:
        if self.num_workers == 1:
            return self.source_datapipe
        return itertools.islice(self.source_datapipe, self.worker, None, self.num_workers)

    def __len__(self) -> int:
        all_len = len(self.source_datapipe)
        return all_len // self.worker + (1 if all_len % self.num_workers > self.num_workers else 0)