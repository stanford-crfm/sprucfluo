from typing import Tuple, Iterator

from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper
import fsspec


@functional_datapipe("open_file_by_fsspec_fancy")
class FancyFSSpecFileOpenerIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Similar to FSSpecFileOpenerIterDataPipe, but it just invokes fsspec.open, and can pass appropriate
    kwargs to it.
    Opens files from input datapipe which contains `fsspec` paths and yields a tuple of
    pathname and opened file stream (functional name: ``open_file_by_fsspec``).

    Args:
        source_datapipe: Iterable DataPipe that provides the pathnames or URLs
        **kwargs: kwargs to pass to fsspec.open

    Example:
        >>> from torchdata.datapipes.iter import FSSpecFileLister
        >>> datapipe = FSSpecFileLister(root=dir_path)
        >>> file_dp = datapipe.open_file_by_fsspec_fancy(mode='rb', compression='infer')
    """

    def __init__(self, source_datapipe: IterDataPipe[str], **kwargs) -> None:
        self.source_datapipe: IterDataPipe[str] = source_datapipe
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for file_uri in self.source_datapipe:
            file = fsspec.open(file_uri, **self.kwargs)
            yield file_uri, StreamWrapper(file.open())

    def __len__(self) -> int:
        return len(self.source_datapipe)


# if "open_file_by_fsspec_fancy" not in IterDataPipe.functions:
#     IterDataPipe.register_datapipe_as_function("open_file_by_fsspec_fancy", FancyFSSpecFileOpenerIterDataPipe)
