import bz2
import gzip
import io
import lzma
import pathlib
import pickle
from collections.abc import Callable
from typing import Any, Dict, Optional, Tuple, Union


class _NoCompression:
    def __init__(self):
        self.open = open
        self.compress = lambda data: data


def _get_compression_from_path(path: Union[str, pathlib.Path]) -> str:
    compressions = {
        ".gz": "gzip",
        ".lzma": "lzma",
        ".bz2": "bz2",
        ".pickle": "no",
    }
    path = pathlib.Path(path)
    if path.suffix not in compressions:
        raise NotImplementedError(f"File extension '{path.suffix}' not supported")
    return compressions[path.suffix]


def _get_compression_library(compression_method: str) -> Any:
    compression_library = {
        "no": _NoCompression(),
        "lzma": lzma,
        "gzip": gzip,
        "bz2": bz2,
    }
    if compression_method not in compression_library:
        raise ValueError(f"Compression method {compression_method} not implemented.")
    return compression_library[compression_method]


def _get_default_kwargs(compression_method: str) -> Dict[str, Any]:
    defaults = {"gzip": {"compresslevel": 1}}
    fallback: Dict[str, Any] = {}  # else mypy complains
    return defaults.get(compression_method, fallback)


def _unpack_compression_args(
    compression: Optional[Union[str, Dict[str, Any]]] = None,
    path: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[str, dict]:
    if compression is not None:
        if isinstance(compression, str):
            return compression, _get_default_kwargs(compression)
        elif isinstance(compression, dict):
            return compression["method"], {
                k: compression[k] for k in compression if k != "method"
            }
        raise ValueError("compression must be either a string or a dict")
    if path is not None:
        # try to find out the compression using the file extension
        compression_method = _get_compression_from_path(path)
        return compression_method, _get_default_kwargs(compression_method)
    raise ValueError("path or compression must not be None.")


def dump_compressed(
    obj: Any,
    path: Union[str, pathlib.Path],
    compression: Optional[Union[str, dict]] = None,
    dump_function: Optional[Callable] = None,
):
    """
    Pickles a model and saves it to the disk. If compression is not specified,
    the compression method will be determined by the file extension.
    :param obj: the object to pickle
    :param path: where to save the object
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to open()
                        of the compression library.
                        Inspired by the pandas.to_csv interface.
    :param dump_function: the function being called to dump the object, can be a custom pickler.
                          Defaults to pickle.dump()
    """
    if dump_function is None:
        dump_function = pickle.dump

    compression_method, kwargs = _unpack_compression_args(compression, path)
    with _get_compression_library(compression_method).open(
        path, mode="wb", **kwargs
    ) as file:
        dump_function(obj, file)


def load_compressed(
    path: Union[str, pathlib.Path], compression: Optional[Union[str, dict]] = None
) -> Any:
    """
    Loads a compressed model.
    :param path: where to load the object from
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to open()
                        of the compression library.
                        Inspired by the pandas.to_csv interface.
    """
    compression_method, kwargs = _unpack_compression_args(compression, path)
    with _get_compression_library(compression_method).open(
        path, mode="rb", **kwargs
    ) as file:
        return pickle.load(file)


def get_pickled_size(
    obj: Any,
    compression: Union[str, dict] = "lzma",
    dump_function: Optional[Callable] = None,
) -> int:
    """
    Returns the size that an object would take on disk if pickled.
    :param obj: the object to pickle
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to
                        compress() of the compression library.
                        Inspired by the pandas.to_csv interface.
    :param dump_function: the function being called to dump the object, can be a custom pickler.
                          Defaults to pickle.dump()
    """
    compression_method, kwargs = _unpack_compression_args(compression, None)
    if dump_function is None:
        dump_function = pickle.dump

    out = io.BytesIO()
    dump_function(obj, out)
    compressed_bytes = _get_compression_library(compression_method).compress(
        out.getvalue(), **kwargs
    )
    return len(compressed_bytes)
