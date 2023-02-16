from pathlib import Path
from typing import Any
import pkg_resources
from sklearn.tree._tree import Tree
from lightgbm import LGBMClassifier
from pickle_compression.pickling import dump_compressed
from _sklearn_tree import dump_function as _sklearn_tree_dump_function
from _lgbm_classifier import dump_function as _lgbm_classifier_dump_function

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"


def pickle_compressed(model: Any, path: Path | str, compression: str | dict = 'lzma'):

    # TODO: match statement?
    if isinstance(model, Tree):
        dump_compressed(model, path, compression, dump_function=_sklearn_tree_dump_function)
    elif isinstance(model, LGBMClassifier):
        dump_compressed(model, path, compression, dump_function=_lgbm_classifier_dump_function)
    else:
        raise NotImplemented(f"Compressed pickling for model of type [{type(model)}] is not supported yet.")