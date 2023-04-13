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

from pathlib import Path
from typing import Any, Optional, Union

import pkg_resources

from slim_trees.pickling import dump_compressed, load_compressed

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"

__all__ = [
    "dump_compressed",
    "load_compressed",
    "dump_sklearn_compressed",
    "dump_lgbm_compressed",
]


def dump_sklearn_compressed(
    model: Any, path: Union[str, Path], compression: Optional[Union[str, dict]] = None
):
    """
    Pickles a model and saves a compressed version to the disk.

    Saves the parameters of the model as int16 and float32 instead of int64 and float64.
    :param model: the model to save
    :param path: where to save the model
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to `open`
                        of the compression library.
                        Options: ["no", "lzma", "gzip", "bz2"]
    """
    from slim_trees.sklearn_tree import dump

    dump_compressed(model, path, compression, dump)


def dump_lgbm_compressed(
    model: Any, path: Union[str, Path], compression: Optional[Union[str, dict]] = None
):
    """
    Pickles a model and saves a compressed version to the disk.

    Saves the parameters of the model as int16 and float32 instead of int64 and float64.
    :param model: the model to save
    :param path: where to save the model
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to `open`
                        of the compression library.
                        Options: ["no", "lzma", "gzip", "bz2"]
    """
    from slim_trees.lgbm_booster import dump

    dump_compressed(model, path, compression, dump)
