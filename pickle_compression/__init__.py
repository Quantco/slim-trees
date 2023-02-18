"""
This module changes the default pickle behavior. Instead of using pickle.dump to pickle an object,
you can create a custom Pickler and change the default pickling behavior for certain classes by
changing the dispatch_table of the pickler.
In this module, a custom pickling behavior for the sklearn Tree class is specified that compresses
the internal values from int64 to int16 and from float64 to float32 (in the value field) and from
float64 to a mixture between float64 and int16 (in the threshold field).
You can pickle a model with custom picklers by specifying dump_function in dump_compressed in the
pickling module.
"""

from pathlib import Path

from typing import Any
import pkg_resources

from pickle_compression.pickling import dump_compressed

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except Exception:
    __version__ = "unknown"


def pickle_compressed(model: Any, path: Path | str, compression: str | dict = "lzma"):
    # depending on the model to be compressed/pickled, choose the respective function
    # this makes sure that we only import the things we need.
    if type(model) in [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "DecisionTreeRegressor",
    ]:
        from _sklearn_tree import pickle_sklearn_compressed

        dump_compressed(
            model, path, compression, dump_function=pickle_sklearn_compressed
        )
    elif str(model) in ["LGBMClassifier()", "LGBMRegressor"]:
        from ._lgbm import dump_function as _lgbm_classifier_dump_function

        dump_compressed(
            model, path, compression, dump_function=_lgbm_classifier_dump_function
        )
    else:
        raise NotImplemented(
            f"Compressed pickling for model of type [{type(model)}] is not supported yet."
        )
