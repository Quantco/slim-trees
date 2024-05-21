import copyreg
import io
import os
import pickle
import re
import sys
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray
from packaging.version import Version

from slim_trees import __version__ as slim_trees_version
from slim_trees.compression_utils import (
    compress_half_int_float_array,
    decompress_half_int_float_array,
    safe_cast,
)
from slim_trees.utils import check_version

try:
    from lightgbm import __version__ as _lightgbm_version
    from lightgbm.basic import Booster

    lightgbm_version = Version(_lightgbm_version)
except ImportError:
    print("LightGBM does not seem to be installed.")
    sys.exit(os.EX_CONFIG)


def dump(model: Any, file: BinaryIO):
    p = pickle.Pickler(file)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dispatch_table[Booster] = _booster_pickle
    p.dump(model)


def dumps(model: Any) -> bytes:
    bytes_io = io.BytesIO()
    dump(model, bytes_io)
    return bytes_io.getvalue()


def _booster_pickle(booster: Booster):
    assert isinstance(booster, Booster)
    reconstructor, args, state = booster.__reduce__()  # type: ignore
    compressed_state = _compress_booster_state(state)
    return _booster_unpickle, (
        reconstructor,
        args,
        (slim_trees_version, compressed_state),
    )


def _booster_unpickle(reconstructor, args, compressed_state):
    version, state = compressed_state
    check_version(version)

    booster = reconstructor(*args)
    decompressed_state = _decompress_booster_state(state)
    booster.__setstate__(decompressed_state)
    return booster


_handle_key_name = (
    "_handle" if lightgbm_version.major == 4 else "handle"  # noqa: PLR2004
)


def _compress_booster_state(state: dict):
    """
    For a given state dictionary, store data in a structured format that can then
    be saved to disk in a way that can be compressed.
    """
    assert isinstance(state, dict)
    compressed_state = {k: v for k, v in state.items() if k != _handle_key_name}
    compressed_state["compressed_handle"] = _compress_booster_handle(
        state[_handle_key_name]
    )
    return compressed_state


def _decompress_booster_state(compressed_state: dict):
    assert isinstance(compressed_state, dict)
    state = {k: v for k, v in compressed_state.items() if k != "compressed_handle"}
    state[_handle_key_name] = _decompress_booster_handle(
        compressed_state["compressed_handle"]
    )
    return state


FRONT_STRING_REGEX = r"(?:\w+(?:=.*)?\n)*\n(?=Tree)"
BACK_STRING_REGEX = r"end of trees(?:\n)+(?:.|\n)*"
TREE_GROUP_REGEX = r"(Tree=\d+\n+)((?:.+\n)*)\n\n"

SPLIT_FEATURE_DTYPE = np.int16
THRESHOLD_DTYPE = np.float64
DECISION_TYPE_DTYPE = np.int8
LEFT_CHILD_DTYPE = np.int16
RIGHT_CHILD_DTYPE = LEFT_CHILD_DTYPE
LEAF_VALUE_DTYPE = np.float64


def _extract_feature(feature_line: str) -> Tuple[str, List[str]]:
    feat_name, values_str = feature_line.split("=")
    return feat_name, values_str.split(" ")


def _validate_feature_lengths(feats_map: dict):
    # features on tree-level
    assert len(feats_map["num_leaves"]) == 1
    assert len(feats_map["num_cat"]) == 1
    assert len(feats_map["is_linear"]) == 1
    assert len(feats_map["shrinkage"]) == 1

    # features on node-level
    num_leaves = int(feats_map["num_leaves"][0])
    assert len(feats_map["split_feature"]) == num_leaves - 1
    assert len(feats_map["threshold"]) == num_leaves - 1
    assert len(feats_map["decision_type"]) == num_leaves - 1
    assert len(feats_map["left_child"]) == num_leaves - 1
    assert len(feats_map["right_child"]) == num_leaves - 1

    # features on leaf-level
    num_leaves = int(feats_map["num_leaves"][0])
    assert len(feats_map["leaf_value"]) == num_leaves


def parse(str_list: Union[List[str], List[Optional[str]]], dtype: DTypeLike):
    if np.can_cast(dtype, np.int64):
        int64_array: NDArray = np.array(str_list, dtype=np.int64)
        return safe_cast(int64_array, dtype)
    assert np.can_cast(dtype, np.float64)
    return np.array(str_list, dtype=dtype)


def _compress_booster_handle(model_string: str) -> Tuple[str, List[dict], str]:
    if not model_string.startswith(f"tree\nversion=v{lightgbm_version.major}"):
        raise ValueError(
            f"Only v{lightgbm_version.major} is supported for the booster string format."
        )

    front_str_match = re.search(FRONT_STRING_REGEX, model_string)
    if front_str_match is None:
        raise ValueError("Could not find front string.")

    # delete tree_sizes line since this messes up the tree parsing by LightGBM if not set correctly
    # todo calculate correct tree_sizes
    front_str = re.sub(r"tree_sizes=(?:\d+ )*\d+\n", "", front_str_match.group())

    back_str_match = re.search(BACK_STRING_REGEX, model_string)
    if back_str_match is None:
        raise ValueError("Could not find back string.")
    back_str = back_str_match.group()
    tree_matches = re.findall(TREE_GROUP_REGEX, model_string)
    trees: List[dict] = []
    for i, tree_match in enumerate(tree_matches):
        tree_name, features_list = tree_match
        _, tree_idx = tree_name.replace("\n", "").split("=")
        assert int(tree_idx) == i

        # extract features -- filter out empty ones
        features: List[str] = [f for f in features_list.split("\n") if "=" in f]
        feats_map: Dict[str, List[str]] = dict(_extract_feature(fl) for fl in features)
        _validate_feature_lengths(feats_map)

        tree_values = {
            "num_leaves": int(feats_map["num_leaves"][0]),
            "num_cat": int(feats_map["num_cat"][0]),
            "split_feature": parse(feats_map["split_feature"], SPLIT_FEATURE_DTYPE),
            "threshold": compress_half_int_float_array(
                parse(feats_map["threshold"], THRESHOLD_DTYPE)
            ),
            "decision_type": parse(feats_map["decision_type"], DECISION_TYPE_DTYPE),
            "left_child": parse(feats_map["left_child"], LEFT_CHILD_DTYPE),
            "right_child": parse(feats_map["right_child"], RIGHT_CHILD_DTYPE),
            "leaf_value": parse(feats_map["leaf_value"], LEAF_VALUE_DTYPE),
            "is_linear": int(feats_map["is_linear"][0]),
            "shrinkage": float(feats_map["shrinkage"][0]),
        }

        # if tree is linear, add additional features
        if int(feats_map["is_linear"][0]):
            # attributes: leaf_features, leaf_coeff, leaf_const, num_features
            # TODO: not all of these attributes might be needed.
            tree_values["num_features"] = parse(feats_map["num_features"], np.int32)
            tree_values["leaf_const"] = parse(feats_map["leaf_const"], LEAF_VALUE_DTYPE)
            tree_values["leaf_features"] = parse(
                [s if s else "-1" for s in feats_map["leaf_features"]],
                np.int16,
            )
            tree_values["leaf_coeff"] = parse(
                [s if s else None for s in feats_map["leaf_coeff"]], np.float64
            )

        # at last
        trees.append(tree_values)

    return front_str, trees, back_str


def _validate_tree_structure(tree: dict):
    assert isinstance(tree, dict)
    if tree.keys() != {
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
    }:
        raise ValueError(
            "Invalid tree structure. Do you use an unsupported LightGBM version or try to load a "
            "model that was pickled with a different version of LightGBM?"
        )


def _decompress_booster_handle(compressed_state: Tuple[str, List[dict], str]) -> str:
    front_str, trees, back_str = compressed_state
    assert isinstance(front_str, str)
    assert isinstance(trees, list)
    assert isinstance(back_str, str)

    handle = front_str
    for i, tree in enumerate(trees):
        _validate_tree_structure(tree)
        is_linear = int(tree["is_linear"])

        num_leaves = len(tree["leaf_value"])
        num_nodes = len(tree["split_feature"])

        tree_str = f"Tree={i}\n"
        tree_str += f"num_leaves={tree['num_leaves']}\nnum_cat={tree['num_cat']}\nsplit_feature="
        tree_str += " ".join([str(x) for x in tree["split_feature"]])
        tree_str += "\nsplit_gain=" + ("0 " * num_nodes)[:-1]
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
        tree_str += "\ninternal_value=" + ("0 " * num_nodes)[:-1]
        tree_str += "\ninternal_weight=" + ("0 " * num_nodes)[:-1]
        tree_str += "\ninternal_count=" + ("0 " * num_nodes)[:-1]
        tree_str += f"\nis_linear={tree['is_linear']}"
        if is_linear:
            tree_str += "\nleaf_const=" + " ".join(str(x) for x in tree["leaf_const"])
            tree_str += "\nnum_features=" + " ".join(
                str(x) for x in tree["num_features"]
            )
            tree_str += "\nleaf_features=" + " ".join(
                "" if f == -1 else str(int(f)) for f in tree["leaf_features"]
            )
            tree_str += "\nleaf_coeff=" + " ".join(
                "" if np.isnan(f) else str(f) for f in tree["leaf_coeff"]
            )

        tree_str += f"\nshrinkage={int(tree['shrinkage'])}"

        tree_str += "\n\n\n"
        handle += tree_str

    handle += back_str
    return handle
