import os
import sys

try:
    from sklearn.tree._tree import Tree
except ImportError:
    print("scikit-learn does not seem to be installed.")
    sys.exit(os.EX_CONFIG)

import copyreg
import pickle
from pathlib import Path
from typing import Any, BinaryIO, Union

import numpy as np

from pickle_compression.pickling import dump_compressed


def pickle_sklearn_compressed(model: Any, file: BinaryIO):
    p = pickle.Pickler(file)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dispatch_table[Tree] = compressed_tree_pickle
    p.dump(model)


def dump_compressed_dtype_reduction(
    model: Any, path: Union[str, Path], compression: Union[str, dict] = "lzma"
):
    """
    Pickles a model and saves a compressed version to the disk.

    Saves the parameters of the model as int16 and float32 instead of int64 and float64.
    :param model: the model to save
    :param path: where to save the model
    :param compression: the compression method used. Either a string or a dict with key 'method' set
                        to the compression method and other key-value pairs are forwarded to `open`
                        of the compression library.
                        Inspired by the pandas.to_csv interface.
    """
    dump_compressed(model, path, compression, pickle_sklearn_compressed)


def compressed_tree_pickle(tree):
    assert isinstance(tree, Tree)
    cls, init_args, state = tree.__reduce__()
    compressed_state = compress_tree_state(state)
    return compressed_tree_unpickle, (cls, init_args, compressed_state)


def compressed_tree_unpickle(cls, init_args, state):
    tree = cls(*init_args)
    decompressed_state = decompress_tree_state(state)
    tree.__setstate__(decompressed_state)
    return tree


def compress_tree_state(state: dict):
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


def decompress_tree_state(state: dict):
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


def _is_in_neighborhood_of_int(arr, iinfo, eps=1e-12):
    """
    Checks if the numbers are around an integer.
    np.abs(arr % 1 - 1) < eps checks if the number is in an epsilon neighborhood on the right side
    of the next int and arr % 1 < eps checks if the number is in an epsilon neighborhood on the left
    side of the next int.
    """
    return (
        (np.minimum(np.abs(arr % 1 - 1), arr % 1) < eps)
        & (arr >= iinfo.min)
        & (arr <= iinfo.max)
    )


def compress_half_int_float_array(a, compression_dtype="int8"):
    """Compress small integer and half-integer floats in a lossless fashion

    Idea:
        If most values in array <a> are small integers or half-integers, we can
        store them as float16, while keeping the rest as float64.

    Technical details:
        - The boolean array (2 * a) % 1 == 0 indicates the integers and half-integers in <a>.
        - int8 can represent integers between np.iinfo('int8').min and np.iinfo('int8').max
    """
    info = np.iinfo(compression_dtype)
    a2 = 2.0 * a
    is_compressible = _is_in_neighborhood_of_int(a2, info)
    not_compressible = np.logical_not(is_compressible)

    a2_compressible = a2[is_compressible].astype(compression_dtype)
    a_incompressible = a[not_compressible]

    state = {
        "is_compressible": is_compressible,
        "a2_compressible": a2_compressible,
        "a_incompressible": a_incompressible,
    }

    return state


def decompress_half_int_float_array(state):
    is_compressible = state["is_compressible"]
    a = np.zeros(len(is_compressible), dtype="float64")
    a[is_compressible] = state["a2_compressible"] / 2.0
    a[np.logical_not(is_compressible)] = state["a_incompressible"]
    return a
