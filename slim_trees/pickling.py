import bz2
import gzip
import io
import lzma
import pathlib
import pickle
from collections.abc import Callable
from typing import Any, BinaryIO, Dict, Optional, Tuple, Union, overload


class _NoCompression:
    def __init__(self):
        self.open = open
        self.compress = lambda data: data

    @staticmethod
    def decompress(data):
        return data


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
    file: Optional[Union[str, pathlib.Path, BinaryIO]] = None,
) -> Tuple[str, dict]:
    if compression is not None:
        if isinstance(compression, str):
            return compression, _get_default_kwargs(compression)
        elif isinstance(compression, dict):
            return compression["method"], {
                k: compression[k] for k in compression if k != "method"
            }
        raise ValueError("compression must be either a string or a dict")
    if file is not None and isinstance(file, (str, pathlib.Path)):
        # try to find out the compression using the file extension
        compression_method = _get_compression_from_path(file)
        return compression_method, _get_default_kwargs(compression_method)
    raise ValueError("File must be a path or compression must not be None.")


@overload
def dump_compressed(
    obj: Any,
    file: BinaryIO,
    compression: Union[str, dict],
    dump_function: Optional[Callable] = None,
): ...


@overload
def dump_compressed(
    obj: Any,
    file: Union[str, pathlib.Path],
    compression: Optional[Union[str, dict]] = None,
    dump_function: Optional[Callable] = None,
): ...


def dump_compressed(
    obj: Any,
    file: Union[str, pathlib.Path, BinaryIO],
    compression: Optional[Union[str, dict]] = None,
    dump_function: Optional[Callable] = None,
):
    """
    Pickles a model and saves it to the disk. If compression is not specified,
    the compression method will be determined by the file extension.
    :param obj: the object to pickle
    :param file: where to save the object, either a path or a file object
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to open()
                        of the compression library.
                        Inspired by the pandas.to_csv interface.
    :param dump_function: the function being called to dump the object, can be a custom pickler.
                          Defaults to pickle.dump()
    """
    if dump_function is None:
        dump_function = pickle.dump

    compression_method, kwargs = _unpack_compression_args(compression, file)
    with _get_compression_library(compression_method).open(
        file, mode="wb", **kwargs
    ) as fd:
        dump_function(obj, fd)


def dumps_compressed(
    obj: Any,
    compression: Optional[Union[str, dict]] = None,
    dump_function: Optional[Callable] = None,
) -> bytes:
    """
    Pickles a model and returns the pickled bytes. If compression is not specified, it won't use
    any.
    :param obj: the object to pickle
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to open()
                        of the compression library. Defaults to 'no' compression.
                        Inspired by the pandas.to_csv interface.
    :param dump_function: the function being called to dump the object, can be a custom pickler.
                          Defaults to pickle.dumps()
    """
    if compression is None:
        compression = "no"

    if dump_function is None:
        dump_function = pickle.dumps

    compression_method, kwargs = _unpack_compression_args(compression, None)
    data_uncompressed = dump_function(obj)
    return _get_compression_library(compression_method).compress(data_uncompressed)


@overload
def load_compressed(
    file: BinaryIO,
    compression: Union[str, dict],
    unpickler_class: type = pickle.Unpickler,
): ...


@overload
def load_compressed(
    file: Union[str, pathlib.Path],
    compression: Optional[Union[str, dict]] = None,
    unpickler_class: type = pickle.Unpickler,
): ...


def load_compressed(
    file: Union[str, pathlib.Path, BinaryIO],
    compression: Optional[Union[str, dict]] = None,
    unpickler_class: type = pickle.Unpickler,
) -> Any:
    """
    Loads a compressed model.
    :param file: where to load the object from, either a path or a file object
    :param compression: the compression method used. Either a string or a dict with key 'method'
                        set to the compression method and other key-value pairs which are forwarded
                        to open() of the compression library.
                        Inspired by the pandas.to_csv interface.
    :param unpickler_class: custom unpickler class derived from pickle.Unpickler.
                            This is useful to restrict possible imports or to allow unpickling
                            when required module or function names have been refactored.
    """
    compression_method, kwargs = _unpack_compression_args(compression, file)
    with _get_compression_library(compression_method).open(
        file, mode="rb", **kwargs
    ) as fd:
        return unpickler_class(fd).load()


def loads_compressed(
    data: bytes,
    compression: Optional[Union[str, dict]] = None,
    unpickler_class: type = pickle.Unpickler,
) -> Any:
    """
    Loads a compressed model.
    :param data: bytes containing the pickled object.
    :param compression: the compression method used. Either a string or a dict with key 'method'
                        set to the compression method and other key-value pairs which are forwarded
                        to open() of the compression library. Defaults to 'no' compression.
                        Inspired by the pandas.to_csv interface.
    :param unpickler_class: custom unpickler class derived from pickle.Unpickler.
                            This is useful to restrict possible imports or to allow unpickling
                            when required module or function names have been refactored.
    """
    if compression is None:
        compression = "no"

    compression_method, kwargs = _unpack_compression_args(compression, None)
    data_uncompressed = _get_compression_library(compression_method).decompress(data)
    return unpickler_class(io.BytesIO(data_uncompressed)).load()


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
