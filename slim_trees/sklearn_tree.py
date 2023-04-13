import os
import sys

from slim_trees import __version__ as slim_trees_version
from slim_trees.compression_utils import (
    compress_half_int_float_array,
    decompress_half_int_float_array,
    safe_cast,
)
from slim_trees.utils import check_version

try:
    from sklearn.tree._tree import Tree
except ImportError:
    print("scikit-learn does not seem to be installed.")
    sys.exit(os.EX_CONFIG)

import copyreg
import pickle
from typing import Any, BinaryIO

import numpy as np


def dump(model: Any, file: BinaryIO):
    p = pickle.Pickler(file)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dispatch_table[Tree] = _tree_pickle
    p.dump(model)


def _tree_pickle(tree):
    assert isinstance(tree, Tree)
    reconstructor, args, state = tree.__reduce__()
    compressed_state = _compress_tree_state(state)
    return _tree_unpickle, (reconstructor, args, (slim_trees_version, compressed_state))


def _tree_unpickle(reconstructor, args, compressed_state):
    version, state = compressed_state
    check_version(version)

    tree = reconstructor(*args)
    decompressed_state = _decompress_tree_state(state)
    tree.__setstate__(decompressed_state)
    return tree


def _compress_tree_state(state: dict):
    """
    Compresses a Tree state.
    :param state: dictionary with 'max_depth', 'node_count', 'nodes', 'values' as keys.
    :return: dictionary with compressed tree state, only with data that is relevant for prediction.
    """
    assert isinstance(state, dict)
    assert state.keys() == {"max_depth", "node_count", "nodes", "values"}
    nodes = state["nodes"]
    # nodes is a numpy array of tuples of the following form
    # (left_child, right_child, feature, threshold, impurity, n_node_samples,
    #  weighted_n_node_samples)
    dtype_child = np.uint16
    dtype_feature = np.uint16
    dtype_threshold = np.float64
    dtype_value = np.float32

    children_left = nodes["left_child"]
    children_right = nodes["right_child"]

    is_leaf = children_left == -1
    # assert that the leaves are the same no matter if you use children_left or children_right
    assert np.array_equal(is_leaf, children_right == -1)

    # feature, threshold and children are irrelevant when leaf

    children_left = safe_cast(children_left[~is_leaf], dtype_child)
    children_right = safe_cast(children_right[~is_leaf], dtype_child)
    features = safe_cast(nodes["feature"][~is_leaf], dtype_feature)
    # value is irrelevant when node not a leaf
    values = safe_cast(state["values"][is_leaf], dtype_value)
    # do lossless compression for thresholds by downcasting half ints (e.g. 5.5, 10.5, ...) to int8
    thresholds = nodes["threshold"][~is_leaf].astype(dtype_threshold)
    thresholds = compress_half_int_float_array(thresholds)

    return {
        "max_depth": state["max_depth"],
        "node_count": state["node_count"],
        "is_leaf": np.packbits(is_leaf),
        "children_left": children_left,
        "children_right": children_right,
        "features": features,
        "thresholds": thresholds,
        "values": values,
    }


def _decompress_tree_state(state: dict):
    """
    Decompresses a Tree state.
    :param state: 'children_left', 'children_right', 'features', 'thresholds', 'values' as keys.
                  'max_depth' and 'node_count' are passed through.
    :return: dictionary with decompressed tree state.
    """
    assert isinstance(state, dict)
    assert state.keys() == {
        "max_depth",
        "node_count",
        "is_leaf",
        "children_left",
        "children_right",
        "features",
        "thresholds",
        "values",
    }
    n_nodes = state["node_count"]

    children_left = np.zeros(n_nodes, dtype=np.int64)
    children_right = np.zeros(n_nodes, dtype=np.int64)
    features = np.zeros(n_nodes, dtype=np.int64)
    thresholds = np.zeros(n_nodes, dtype=np.float64)
    # same shape as values but with all nodes instead of only the leaves
    values = np.zeros((n_nodes, *state["values"].shape[1:]), dtype=np.float64)

    is_leaf = np.unpackbits(state["is_leaf"], count=n_nodes).astype("bool")
    children_left[~is_leaf] = state["children_left"]
    children_left[is_leaf] = -1
    children_right[~is_leaf] = state["children_right"]
    children_right[is_leaf] = -1
    features[~is_leaf] = state["features"]
    features[is_leaf] = -2  # feature of leaves is -2
    thresholds[~is_leaf] = decompress_half_int_float_array(state["thresholds"])
    thresholds[is_leaf] = -2.0  # threshold of leaves is -2
    values[is_leaf] = state["values"]

    dtype = np.dtype(
        [
            ("left_child", "<i8"),
            ("right_child", "<i8"),
            ("feature", "<i8"),
            ("threshold", "<f8"),
            ("impurity", "<f8"),
            ("n_node_samples", "<i8"),
            ("weighted_n_node_samples", "<f8"),
        ]
    )
    nodes = np.zeros(n_nodes, dtype=dtype)
    nodes["left_child"] = children_left
    nodes["right_child"] = children_right
    nodes["feature"] = features
    nodes["threshold"] = thresholds

    return {
        "max_depth": state["max_depth"],
        "node_count": state["node_count"],
        "nodes": nodes,
        "values": values,
    }
