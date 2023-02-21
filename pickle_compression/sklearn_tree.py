import os
import sys

from pickle_compression.compression_utils import (
    compress_half_int_float_array,
    decompress_half_int_float_array,
)

try:
    from sklearn.tree._tree import Tree
except ImportError:
    print("scikit-learn does not seem to be installed.")
    sys.exit(os.EX_CONFIG)

import copyreg
import pickle
from typing import Any, BinaryIO

import numpy as np


def dump_sklearn(model: Any, file: BinaryIO):
    p = pickle.Pickler(file)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dispatch_table[Tree] = _tree_pickle
    p.dump(model)


def _tree_pickle(tree):
    assert isinstance(tree, Tree)
    reconstructor, args, state = tree.__reduce__()
    compressed_state = _compress_tree_state(state)
    return _tree_unpickle, (reconstructor, args, compressed_state)


def _tree_unpickle(reconstructor, args, state):
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
    dtype_child = np.int16
    dtype_feature = np.int16
    dtype_threshold = np.float64
    dtype_value = np.float32

    children_left = nodes["left_child"].astype(dtype_child)
    children_right = nodes["right_child"].astype(dtype_child)

    is_leaf = children_left == -1
    is_not_leaf = np.logical_not(is_leaf)
    # assert that the leaves are the same no matter if you use children_left or children_right
    assert np.array_equal(is_leaf, children_right == -1)

    # feature, threshold and children are irrelevant when leaf
    # don't omit children_left -1 values because they are needed to identify leaves
    children_right = children_right[is_not_leaf]
    features = nodes["feature"][is_not_leaf].astype(dtype_feature)
    # value is irrelevant when node not a leaf
    values = state["values"][is_leaf].astype(dtype_value)
    # do lossless compression for thresholds by downcasting half ints (e.g. 5.5, 10.5, ...) to int8
    thresholds = nodes["threshold"][is_not_leaf].astype(dtype_threshold)
    thresholds = compress_half_int_float_array(thresholds)

    return {
        "max_depth": state["max_depth"],
        "node_count": state["node_count"],
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
        "children_left",
        "children_right",
        "features",
        "thresholds",
        "values",
    }
    children_left = state["children_left"].astype(np.int64)
    n_edges = len(children_left)
    is_leaf = children_left == -1
    is_not_leaf = np.logical_not(is_leaf)

    children_right = np.zeros(n_edges, dtype=np.int64)
    features = np.zeros(n_edges, dtype=np.int64)
    thresholds = np.zeros(n_edges, dtype=np.float64)
    # same shape as values but with all edges instead of only the leaves
    values = np.zeros((n_edges, *state["values"].shape[1:]), dtype=np.float64)

    children_right[is_not_leaf] = state["children_right"]
    children_right[is_leaf] = -1
    features[is_not_leaf] = state["features"]
    features[is_leaf] = -2  # feature of leaves is -2
    thresholds[is_not_leaf] = decompress_half_int_float_array(state["thresholds"])
    thresholds[is_leaf] = -2  # threshold of leaves is -2
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
    nodes = np.zeros(n_edges, dtype=dtype)
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
