import copyreg
import os
import pickle
import re
import sys
from typing import Any, BinaryIO, List, Optional, Tuple

import numpy as np
import pandas as pd

from slim_trees import __version__ as slim_trees_version
from slim_trees.utils import check_version, df_to_pq_bytes, pq_bytes_to_df

FRONT_STRING_REGEX = r"(?:\w+(?:=.*)?\n)*\n(?=Tree)"
BACK_STRING_REGEX = r"end of trees(?:\n)+(?:.|\n)*"
TREE_GROUP_REGEX = r"(Tree=\d+\n+)((?:.+\n)*)\n\n"

SPLIT_FEATURE_DTYPE = np.int16
THRESHOLD_DTYPE = np.float64
DECISION_TYPE_DTYPE = np.int8
LEFT_CHILD_DTYPE = np.int16
RIGHT_CHILD_DTYPE = LEFT_CHILD_DTYPE
LEAF_VALUE_DTYPE = np.float64

try:
    from lightgbm.basic import Booster
except ImportError:
    print("LightGBM does not seem to be installed.")
    sys.exit(os.EX_CONFIG)


def dump_lgbm(model: Any, file: BinaryIO):
    p = pickle.Pickler(file)
    p.dispatch_table = copyreg.dispatch_table.copy()
    p.dispatch_table[Booster] = _booster_pickle
    p.dump(model)


def _booster_pickle(booster: Booster):
    assert isinstance(booster, Booster)
    reconstructor, args, state = booster.__reduce__()
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


def _extract_feature(feature_line: str) -> Tuple[str, List[str]]:
    feat_name, values_str = feature_line.split("=")
    return feat_name, values_str.split(" ")


def parse(str_list, dtype):
    return np.array(str_list, dtype=dtype)


def _compress_booster_handle(
    model_string: str,
) -> Tuple[str, bytes, bytes, bytes, Optional[bytes], str]:
    if not model_string.startswith("tree\nversion=v3"):
        raise ValueError("Only v3 is supported for the booster string format.")

    front_str_match = re.search(FRONT_STRING_REGEX, model_string)
    if front_str_match is None:
        raise ValueError("Could not find front string.")
    # todo calculate correct tree_sizes
    front_str = re.sub(r"tree_sizes=(?:\d+ )*\d+\n", "", front_str_match.group())

    back_str_match = re.search(BACK_STRING_REGEX, model_string)
    if back_str_match is None:
        raise ValueError("Could not find back string.")
    back_str = back_str_match.group()

    tree_matches = re.findall(TREE_GROUP_REGEX, model_string)
    node_features: List[dict] = []
    leaf_values: List[dict] = []
    trees: List[dict] = []
    linear_values: List[dict] = []
    for i, tree_match in enumerate(tree_matches):
        tree_name, features_list = tree_match
        _, tree_idx = tree_name.replace("\n", "").split("=")

        # extract features -- filter out empty ones
        features = [f for f in features_list.split("\n") if "=" in f]
        feats_map = dict(_extract_feature(fl) for fl in features)

        assert int(tree_idx) == i
        assert len(feats_map["num_leaves"]) == 1
        assert len(feats_map["num_cat"]) == 1
        assert len(feats_map["is_linear"]) == 1
        assert len(feats_map["shrinkage"]) == 1

        # length = 1
        trees.append(
            {
                "tree_idx": int(tree_idx),
                "num_leaves": int(feats_map["num_leaves"][0]),
                "num_cat": int(feats_map["num_cat"][0]),
                "is_linear": int(feats_map["is_linear"][0]),
                "shrinkage": float(feats_map["shrinkage"][0]),
            }
        )

        # length = num_inner_nodes = num_leaves - 1
        node_features.append(
            {
                "tree_idx": int(tree_idx),
                # all the upcoming attributes have length num_leaves - 1
                "split_feature": parse(feats_map["split_feature"], SPLIT_FEATURE_DTYPE),
                "threshold": parse(feats_map["threshold"], THRESHOLD_DTYPE),
                "decision_type": parse(feats_map["decision_type"], DECISION_TYPE_DTYPE),
                "left_child": parse(feats_map["left_child"], LEFT_CHILD_DTYPE),
                "right_child": parse(feats_map["right_child"], RIGHT_CHILD_DTYPE),
            }
        )

        # length = num_leaves
        leaf_values.append(
            {
                "tree_idx": int(tree_idx),
                "leaf_value": parse(feats_map["leaf_value"], LEAF_VALUE_DTYPE),
            }
        )

        # length = sum_l=0^{num_leaves} {num_features(l)}
        # attributes: leaf_features, leaf_coeff, leaf_const, num_features\
        # TODO: some of these attributes, e.g. leaf_const, might not be needed
        if "leaf_features" in feats_map:
            leaf_values[-1]["leaf_const"] = parse(
                feats_map["leaf_const"], LEAF_VALUE_DTYPE
            )
            leaf_values[-1]["num_features"] = parse(feats_map["num_features"], np.int32)

            linear_values.append(
                {
                    "tree_idx": int(tree_idx),
                    "leaf_features": parse(
                        [s if s else -1 for s in feats_map["leaf_features"]],
                        np.int16,
                    ),
                    "leaf_coeff": parse(
                        [s if s else None for s in feats_map["leaf_coeff"]], np.float64
                    ),
                }
            )

    tree_value_bytes = df_to_pq_bytes(pd.DataFrame(trees))

    nodes_df = pd.DataFrame(node_features)
    node_values_bytes = df_to_pq_bytes(
        nodes_df.explode(
            [
                "split_feature",
                "threshold",
                "decision_type",
                "left_child",
                "right_child",
            ]
        )
    )

    leaf_values_bytes = df_to_pq_bytes(
        pd.DataFrame(leaf_values).explode(
            ["leaf_value"] + (["leaf_const", "num_features"] if linear_values else [])
        )
    )

    linear_values_bytes = None
    if linear_values:
        linear_values_bytes = df_to_pq_bytes(
            pd.DataFrame(linear_values).explode(["leaf_features", "leaf_coeff"])
        )

    return (
        front_str,
        tree_value_bytes,
        node_values_bytes,
        leaf_values_bytes,
        linear_values_bytes,
        back_str,
    )


def _decompress_booster_handle(
    compressed_state: Tuple[str, bytes, bytes, bytes, bytes, str]
) -> str:
    (
        front_str,
        trees_df_bytes,
        nodes_df_bytes,
        leaf_value_bytes,
        linear_values_bytes,
        back_str,
    ) = compressed_state
    assert type(front_str) == str
    assert type(back_str) == str

    trees_df = pq_bytes_to_df(trees_df_bytes)
    nodes_df = pq_bytes_to_df(nodes_df_bytes).groupby("tree_idx").agg(lambda x: list(x))
    leaf_values_df = (
        pq_bytes_to_df(leaf_value_bytes).groupby("tree_idx").agg(lambda x: list(x))
    )

    # merge trees_df, nodes_df, and leaf_values_df on tree_idx
    trees_df = trees_df.merge(nodes_df, on="tree_idx")
    trees_df = trees_df.merge(leaf_values_df, on="tree_idx")
    if linear_values_bytes is not None:
        linear_values_df = (
            pq_bytes_to_df(linear_values_bytes)
            .groupby("tree_idx")
            .agg(lambda x: list(x))
        )
        trees_df = trees_df.merge(linear_values_df, on="tree_idx")

    tree_strings = [front_str]

    for i, tree in trees_df.iterrows():
        num_leaves = int(tree["num_leaves"])
        num_nodes = num_leaves - 1

        # add the appropriate block if those values are present
        if tree["is_linear"]:
            linear_str = f"""
leaf_const={" ".join(str(x) for x in tree['leaf_const'])}
num_features={" ".join(str(x) for x in tree['num_features'])}
leaf_features={" ".join(["" if f == -1 else str(int(f)) for f in tree['leaf_features']])}
leaf_coeff={" ".join(["" if np.isnan(f) else str(f) for f in tree['leaf_coeff']])}"""
        else:
            linear_str = ""

        tree_strings.append(
            f"""Tree={i}
num_leaves={int(tree["num_leaves"])}
num_cat={tree['num_cat']}
split_feature={' '.join([str(x) for x in tree["split_feature"]])}
split_gain={("0" * num_nodes)[:-1]}
threshold={' '.join([str(x) for x in tree['threshold']])}
decision_type={' '.join([str(x) for x in tree["decision_type"]])}
left_child={" ".join([str(x) for x in tree["left_child"]])}
right_child={" ".join([str(x) for x in tree["right_child"]])}
leaf_value={" ".join([str(x) for x in tree["leaf_value"]])}
leaf_weight={("0 " * num_leaves)[:-1]}
leaf_count={("0 " * num_leaves)[:-1]}
internal_value={("0 " * num_nodes)[:-1]}
internal_weight={("0 " * num_nodes)[:-1]}
internal_count={("0 " * num_nodes)[:-1]}
is_linear={tree['is_linear']}{linear_str}
shrinkage={tree['shrinkage']}


"""
        )

    tree_strings.append(back_str)

    return "".join(tree_strings)
