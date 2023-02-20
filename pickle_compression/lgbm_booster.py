import copyreg
import os
import pickle
import sys
from typing import Any, BinaryIO

try:
    from lightgbm.basic import Booster
except ImportError:
    print("LightGBM does not seem to be installed.")
    sys.exit(os.EX_CONFIG)


def dump_lgbm(model: Any, file: BinaryIO):
    p = pickle.Pickler(file)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dispatch_table[Booster] = _compressed_booster_pickle
    p.dump(model)


def _compressed_booster_pickle(booster: Booster):
    assert isinstance(booster, Booster)

    # retrieve
    cls, init_args, state = booster.__reduce__()

    # extract state information
    """
    Debug: This is a dataframe that contains one row per tree and some metadata we could use
    for navigating the dictionary.
    columns are :
    ['tree_index', 'node_depth', 'node_index', 'left_child', 'right_child',
       'parent_index', 'split_feature', 'split_gain', 'threshold',
       'decision_type', 'missing_direction', 'missing_type', 'value',
       'weight','count'
    ]
    """

    # this dataframe contains meta-information about each tree (maybe not important/redundant)
    # lgbm_booster.trees_to_dataframe()

    # this contains the same information as the .model/.txt model files
    # keys: (['name', 'version', 'num_class', 'num_tree_per_iteration', 'label_index',
    # 'max_feature_idx', 'objective', 'average_output', 'feature_names', 'monotone_constraints',
    # 'feature_infos', 'tree_info', 'feature_importances', 'pandas_categorical']
    booster.dump_model()

    # transform and compress state
    compressed_state = _compress_lgbm_state(state)

    # return function to unpickle again
    return _compressed_booster_unpickle, (cls, init_args, compressed_state)


def _compressed_booster_unpickle(cls, init_args, compressed_state):
    _class, base, state = init_args  # unpack
    state = {"model_str": compressed_state}
    cls(_class, base, state)
    state = _decompress_lgbm_state(compressed_state)
    model_string = state["handle"]
    # https://github.com/microsoft/LightGBM/issues/5370
    # currently it's not possible to de-serialize out of the JSON/dict again
    # tree.__setstate__(decompressed_state)
    # TODO: find a way to create a Booster back again from it's state representation
    return Booster(model_str=model_string)


def _compress_lgbm_state(state: dict):
    """
    For a given state dictionary, store data in a structured format that can then
    be saved to disk in a way that can be compressed.
    """
    return state  # TODO: actually _do_ something


def _decompress_lgbm_state(compressed_state):
    # TODO: revert what has been done in _compress_lgbm_state
    return compressed_state
