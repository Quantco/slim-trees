"""
This module changes the default pickle behavior. Instead of using pickle.dump to pickle an object,
you can create a custom Pickler and change the default pickling behavior for certain classes by
changing the dispatch_table of the pickler.
TODO: Update this docstring
In this module, a custom pickling behavior for the sklearn Tree class is specified that compresses
the internal values from int64 to int16 and from float64 to float32 (in the value field) and from
float64 to a mixture between float64 and int16 (in the threshold field).
You can pickle a model with custom picklers by specifying dump_function in dump_compressed in the
pickling module.
"""

import importlib.metadata
import warnings
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union, overload

from slim_trees.pickling import (
    dump_compressed,
    dumps_compressed,
    load_compressed,
    loads_compressed,
)

try:
    __version__ = importlib.metadata.version(__name__)
except Exception as e:
    warnings.warn(f"Could not determine version of {__name__}", stacklevel=1)
    warnings.warn(str(e), stacklevel=1)
    __version__ = "unknown"

__all__ = [
    "dump_compressed",
    "dumps_compressed",
    "load_compressed",
    "loads_compressed",
    "dump_sklearn_compressed",
    "dumps_sklearn_compressed",
    "dump_lgbm_compressed",
    "dumps_lgbm_compressed",
]


@overload
def dump_sklearn_compressed(
    model: Any,
    file: BinaryIO,
    compression: Union[str, dict],
): ...


@overload
def dump_sklearn_compressed(
    model: Any,
    file: Union[str, Path],
    compression: Optional[Union[str, dict]] = None,
): ...


def dump_sklearn_compressed(
    model: Any,
    file: Union[str, Path, BinaryIO],
    compression: Optional[Union[str, dict]] = None,
):
    """
    Pickles a model and saves a compressed version to the disk.

    Saves the parameters of the model as int16 and float32 instead of int64 and float64.
    :param model: the model to save
    :param file: where to save the model, either a path or a file object
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to `open`
                        of the compression library.
                        Options: ["no", "lzma", "gzip", "bz2"]
    """
    from slim_trees.sklearn_tree import dump

    dump_compressed(model, file, compression, dump)  # type: ignore


def dumps_sklearn_compressed(
    model: Any, compression: Optional[Union[str, dict]] = None
) -> bytes:
    """
    Pickles a model and returns the saved object as bytes.

    Saves the parameters of the model as int16 and float32 instead of int64 and float64.
    :param model: the model to save
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to `open`
                        of the compression library.
                        Options: ["no", "lzma", "gzip", "bz2"]
    """
    from slim_trees.sklearn_tree import dumps

    return dumps_compressed(model, compression, dumps)


@overload
def dump_lgbm_compressed(
    model: Any,
    file: BinaryIO,
    compression: Union[str, dict],
): ...


@overload
def dump_lgbm_compressed(
    model: Any,
    file: Union[str, Path],
    compression: Optional[Union[str, dict]] = None,
): ...


def dump_lgbm_compressed(
    model: Any,
    file: Union[str, Path, BinaryIO],
    compression: Optional[Union[str, dict]] = None,
):
    """
    Pickles a model and saves a compressed version to the disk.

    Saves the parameters of the model as int16 and float32 instead of int64 and float64.
    :param model: the model to save
    :param file: where to save the model, either a path or a file object
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to `open`
                        of the compression library.
                        Options: ["no", "lzma", "gzip", "bz2"]
    """
    from slim_trees.lgbm_booster import dump

    dump_compressed(model, file, compression, dump)  # type: ignore


def dumps_lgbm_compressed(
    model: Any, compression: Optional[Union[str, dict]] = None
) -> bytes:
    """
    Pickles a model and returns the saved object as bytes.

    Saves the parameters of the model as int16 and float32 instead of int64 and float64.
    :param model: the model to save
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to `open`
                        of the compression library.
                        Options: ["no", "lzma", "gzip", "bz2"]
    """
    from slim_trees.lgbm_booster import dumps

    return dumps_compressed(model, compression, dumps)
