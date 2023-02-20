import copyreg
import os
import pickle
import sys
from typing import Any, BinaryIO, List, Tuple

import numpy as np
from compression_utils import (
    compress_half_int_float_array,
    decompress_half_int_float_array,
)

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
    reconstructor, args, state = booster.__reduce__()
    compressed_state = _compress_booster_state(state)
    return _compressed_booster_unpickle, (reconstructor, args, compressed_state)


def _compressed_booster_unpickle(reconstructor, args, compressed_state):
    booster = reconstructor(*args)
    decompressed_state = _decompress_booster_state(compressed_state)
    booster.__setstate__(decompressed_state)
    return booster


def _compress_booster_state(state: dict):
    """
    For a given state dictionary, store data in a structured format that can then
    be saved to disk in a way that can be compressed.
    """
    assert type(state) == dict
    compressed_state = {k: v for k, v in state.items() if k != "handle"}
    compressed_state["compressed_handle"] = _compress_booster_handle(state["handle"])
    return compressed_state


def _decompress_booster_state(compressed_state: dict):
    assert type(compressed_state) == dict
    state = {k: v for k, v in compressed_state.items() if k != "compressed_handle"}
    state["handle"] = _decompress_booster_handle(compressed_state["compressed_handle"])
    return state


def _compress_booster_handle(model_string: str) -> Tuple[str, List[dict], str]:
    if not model_string.startswith("tree\nversion=v3"):
        raise ValueError("Only v3 is supported for the booster string format.")
    print("Warning: _compress_booster_handle is not implemented")
    tree = {
        "num_cat": 0,
        "split_feature": np.array([2132, 44, 142, 182, 2132, 15, 261]),
        "threshold": compress_half_int_float_array(
            np.array([0.5, 1.522, 0.5, 0.5, 0.5, 0.5, 0.5])
        ),
        "decision_type": np.array([2, 2, 10, 10, 2, 2]),
        "left_child": np.array([4, 2, -2, -3, 6, -6, -1]),
        "right_child": np.array([1, 3, -4, -5, 5, -7, -8]),
        "leaf_value": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "is_linear": 0,
        "shrinkage": 0.05,
    }

    # todo put this somewhere in the parser where it makes sense
    # assert model_string[tree]["num_leaves"] == len(tree["leaf_value"])
    return "", [tree], ""


def _decompress_booster_handle(compressed_state: Tuple[str, List[dict], str]) -> str:
    front_str, trees, back_str = compressed_state
    assert type(front_str) == str
    assert type(trees) == list
    assert type(back_str) == str

    handle = front_str

    for i, tree in enumerate(trees):
        assert type(tree) == dict
        assert tree.keys() == {
            "num_leaves",
            "num_cat",
            "split_feature",
            "threshold",
            "decision_type",
            "left_child",
            "right_child",
            "leaf_value",
            "is_linear",
            "shrinkage",
        }

        num_leaves = len(tree["leaf_value"])
        num_nodes = len(tree["split_feature"])

        tree_str = f"Tree={i}\n"
        tree_str += f"num_leaves={tree['num_leaves']}\nnum_cat={tree['num_cat']}\nsplit_feature="
        tree_str += " ".join([str(x) for x in tree["split_feature"]])
        tree_str += "\nsplit_gain=" + ("0.0 " * num_nodes)[:-1]
        threshold = decompress_half_int_float_array(tree["threshold"])
        tree_str += "\nthreshold=" + " ".join([str(x) for x in threshold])
        tree_str += "\ndecision_type=" + " ".join(
            [str(x) for x in tree["decision_type"]]
        )
        tree_str += "\nleft_child=" + " ".join([str(x) for x in tree["left_child"]])
        tree_str += "\nright_child=" + " ".join([str(x) for x in tree["right_child"]])
        tree_str += "\nleaf_value=" + " ".join([str(x) for x in tree["leaf_value"]])
        tree_str += "\nleaf_weight=" + ("0 " * num_leaves)[:-1]
        tree_str += "\nleaf_count=" + ("0 " * num_leaves)[:-1]
        tree_str += "\ninternal_value=" + ("0.0 " * num_nodes)[:-1]
        tree_str += "\ninternal_weight=" + ("0 " * num_nodes)[:-1]
        tree_str += "\ninternal_count=" + ("0 " * num_nodes)[:-1]
        tree_str += (
            f"\nis_linear{tree['is_linear']}\nshrinkage={tree['shrinkage']}\n\n\n"
        )

        handle += tree_str
    handle += "end of trees\n\n" + back_str
    return handle
