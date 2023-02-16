import copyreg
import pickle
from typing import BinaryIO

from lightgbm import LGBMClassifier

_target_type = LGBMClassifier

def dump_function(model: _target_type, file: BinaryIO):
    p = pickle.Pickler(file)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dispatch_table[_target_type] = _compressed_lgbm_pickle
    p.dump(model)

def _compressed_lgbm_pickle(lgbm_classifier: _target_type):
    assert isinstance(lgbm_classifier, _target_type)

    # retrieve 
    cls, init_args, _ = lgbm_classifier.booster_.__reduce__()

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
    tree_df = lgbm_classifier.booster_.trees_to_dataframe()
    # this contains the same information as the .model/.txt model files
    # keys: (['name', 'version', 'num_class', 'num_tree_per_iteration', 'label_index', 'max_feature_idx', 'objective', 'average_output', 'feature_names', 'monotone_constraints', 'feature_infos', 'tree_info', 'feature_importances', 'pandas_categorical']
    dump_dict = lgbm_classifier.booster_.dump_model()

    # transform and compress state
    compressed_state = _compress_lgbm_state(dump_dict)

    # return function to unpickle again
    return _compressed_lgbm_unpickle, (cls, init_args, compressed_state)

def _compress_lgbm_state(state):
    """
    For a given state dictionary, store data in a structured format that can then
    be saved to disk in a way that can be compressed.
    """
    return state # TODO: actually do something

def _decompress_lgbm_state(compressed_state):
    return compressed_state

def _compressed_lgbm_unpickle(cls, init_args, compressed_state):
    tree = cls(*init_args)
    decompressed_state = _decompress_lgbm_state(compressed_state)
    # https://github.com/microsoft/LightGBM/issues/5370
    # currently it's not possible to de-serialize out of the JSON/dict again
    # tree.__setstate__(decompressed_state)
    return tree