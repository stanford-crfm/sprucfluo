import os
from typing import Tuple, Iterator

from torch.utils.data import functional_datapipe, IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper
import fsspec
import fsspec.compression
import fsspec.utils


@functional_datapipe("open_file_by_fsspec_fancy")
class FancyFSSpecFileOpenerIterDataPipe(IterDataPipe[Tuple[str, StreamWrapper]]):
    r"""
    Similar to FSSpecFileOpenerIterDataPipe, but it just invokes fsspec.open, and can pass appropriate
    kwargs to it.
    Opens files from input datapipe which contains `fsspec` paths and yields a tuple of
    pathname and opened file stream (functional name: ``open_file_by_fsspec_fancy``).
    The pathname is munged to be only the file name of the final path, without all of the uri fanciness. Any compression
    extensions are removed from the pathname if the file is being decompressed.

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
        self.kwargs = kwargs.copy()

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        for file_uri in self.source_datapipe:
            file = fsspec.open(file_uri, **self.kwargs)
            # this is similar to the logic in compression=infer in fsspec.open, but we just
            # want to remove the compression extension from the path if applicable
            path = file.path
            if file.compression is not None:
                compr = fsspec.utils.infer_compression(path)
                if compr == file.compression:
                    # strip the compression ext from the path
                    path = os.path.splitext(path)[0]

            yield path, StreamWrapper(file.open())

    def __len__(self) -> int:
        return len(self.source_datapipe)



