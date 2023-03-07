import copyreg
import os
import pickle
import re
import sys
from typing import Any, BinaryIO, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from slim_trees.utils import pq_bytes_to_df, pyarrow_table_to_bytes

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
    return _booster_unpickle, (reconstructor, args, compressed_state)


def _booster_unpickle(reconstructor, args, compressed_state):
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


def _compress_booster_handle(model_string: str) -> Tuple[str, bytes, bytes, bytes, str]:
    if not model_string.startswith("tree\nversion=v3"):
        raise ValueError("Only v3 is supported for the booster string format.")
    FRONT_STRING_REGEX = r"(?:\w+(?:=.*)?\n)*\n(?=Tree)"
    BACK_STRING_REGEX = r"end of trees(?:\n)+(?:.|\n)*"
    TREE_GROUP_REGEX = r"(Tree=\d+\n+)((?:.+\n)*)\n\n"

    def _extract_feature(feature_line):
        feat_name, values_str = feature_line.split("=")
        return feat_name, values_str.split(" ")

    front_str_match = re.search(FRONT_STRING_REGEX, model_string)
    if front_str_match is None:
        raise ValueError("Could not find front string.")
    front_str = front_str_match.group()
    # delete tree_sizes line since this messes up the tree parsing by LightGBM if not set correctly
    # todo calculate correct tree_sizes
    front_str = re.sub(r"tree_sizes=(?:\d+ )*\d+\n", "", front_str)

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
        assert int(tree_idx) == i

        # extract features -- filter out empty ones
        features = [f for f in features_list.split("\n") if "=" in f]
        feats_map = dict(_extract_feature(fl) for fl in features)

        def parse(str_list, dtype):
            return np.array(str_list, dtype=dtype)

        split_feature_dtype = np.int16
        threshold_dtype = np.float64
        decision_type_dtype = np.int8
        left_child_dtype = np.int16
        right_child_dtype = left_child_dtype
        leaf_value_dtype = np.float64
        assert len(feats_map["num_leaves"]) == 1
        assert len(feats_map["num_cat"]) == 1
        assert len(feats_map["is_linear"]) == 1
        assert len(feats_map["shrinkage"]) == 1

        """
        strategy: we have two datastructures: one on tree_level and one on node_level
        here, this looks like just splitting the features into two dicts
        but one of them can be "exploded" later (node level) while the tree level is for meta information
        """

        # length = 1
        trees.append(
            {
                "tree_idx": int(tree_idx),
                "num_leaves": int(feats_map["num_leaves"][0]),
                "num_cat": int(feats_map["num_cat"][0]),
                # "last_leaf_value": parse(feats_map["leaf_value"], leaf_value_dtype)[-1],
                "is_linear": int(feats_map["is_linear"][0]),
                "shrinkage": float(feats_map["shrinkage"][0]),
            }
        )

        # length = num_inner_nodes = num_leaves - 1
        node_features.append(
            {
                "tree_idx": int(
                    tree_idx
                ),  # TODO: this is new, have to recover this as well
                "node_idx": list(range(int(feats_map["num_leaves"][0]) - 1)),
                # all the upcoming attributes have length num_leaves - 1
                "split_feature": parse(feats_map["split_feature"], split_feature_dtype),
                "threshold": parse(feats_map["threshold"], threshold_dtype),
                "decision_type": parse(feats_map["decision_type"], decision_type_dtype),
                "left_child": parse(feats_map["left_child"], left_child_dtype),
                "right_child": parse(feats_map["right_child"], right_child_dtype),
                # "leaf_value": parse(feats_map["leaf_value"], leaf_value_dtype)[:-1],
            }
        )

        # length = num_leaves
        leaf_values.append(
            {
                "tree_idx": int(tree_idx),
                "leaf_value": parse(feats_map["leaf_value"], leaf_value_dtype),
            }
        )

        # length = sum_l=0^{num_leaves} {num_features(l)}
        # attributes: leaf_features, leaf_coeff
        linear_values.append(
            {

            }
        )

    trees_df = pd.DataFrame(trees)
    trees_table = pa.Table.from_pandas(trees_df)
    trees_df_bytes = pyarrow_table_to_bytes(trees_table)

    # transform nodes_df s.t. each feature is a column
    # pyarrow table from list of dicts
    nodes_df = pd.DataFrame(node_features)
    nodes_df = nodes_df.explode(
        [
            "node_idx",
            "split_feature",
            "threshold",
            "decision_type",
            "left_child",
            "right_child",
            # "leaf_value",
        ]
    )
    nodes_table = pa.Table.from_pandas(nodes_df)

    nodes_df_bytes = pyarrow_table_to_bytes(nodes_table)
    leaf_values_df = pd.DataFrame(leaf_values).explode(["leaf_value"])
    leaf_values_table = pa.Table.from_pandas(leaf_values_df)
    leaf_values_bytes = pyarrow_table_to_bytes(leaf_values_table)

    return front_str, trees_df_bytes, nodes_df_bytes, leaf_values_bytes, back_str


# @profile
def _decompress_booster_handle(
        compressed_state: Tuple[str, bytes, bytes, bytes, str]
) -> str:
    (
        front_str,
        trees_df_bytes,
        nodes_df_bytes,
        leaf_value_bytes,
        back_str,
    ) = compressed_state
    assert type(front_str) == str
    assert type(back_str) == str

    trees_df = pq_bytes_to_df(trees_df_bytes)
    nodes_df = pq_bytes_to_df(nodes_df_bytes).groupby("tree_idx").agg(lambda x: list(x))
    leaf_values_df = (
        pq_bytes_to_df(leaf_value_bytes).groupby("tree_idx")["leaf_value"].apply(list)
    )
    # merge trees_df, nodes_df, and leaf_values_df on tree_idx
    trees_df = trees_df.merge(nodes_df, on="tree_idx")
    trees_df = trees_df.merge(leaf_values_df, on="tree_idx")

    # handle = front_str

    tree_strings = [front_str]

    # TODO: directly go over trees and nodes
    for i, tree in trees_df.iterrows():
        # """
        # assert type(tree) == dict
        # assert tree.keys() == {
        #     "num_leaves",
        #     "num_cat",
        #     "split_feature",
        #     "threshold",
        #     "decision_type",
        #     "left_child",
        #     "right_child",
        #     "leaf_value",
        #     "is_linear",
        #     "shrinkage",
        # }
        # """

        num_leaves = int(tree["num_leaves"])
        num_nodes = num_leaves - 1

        tree_strings.append(f"""Tree={i}
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
is_linear={tree['is_linear']}
shrinkage={tree['shrinkage']}


""")
        # TODO: remove old code
        # num_leaves = int(tree["num_leaves"])
        # num_nodes = num_leaves - 1  # len(tree["split_feature"])
        #
        # tree_str = f"Tree={i}\n"
        # tree_str += (
        #     f"num_leaves={num_leaves}\nnum_cat={tree['num_cat']}\nsplit_feature="
        # )
        # tree_str += " ".join([str(x) for x in tree["split_feature"]])
        # tree_str += "\nsplit_gain=" + ("0 " * num_nodes)[:-1]
        # threshold = tree[
        #     "threshold"
        # ]  # decompress_half_int_float_array(tree["threshold"])
        # tree_str += "\nthreshold=" + " ".join([str(x) for x in threshold])
        # tree_str += "\ndecision_type=" + " ".join(
        #     [str(x) for x in tree["decision_type"]]
        # )
        # tree_str += "\nleft_child=" + " ".join([str(x) for x in tree["left_child"]])
        # tree_str += "\nright_child=" + " ".join([str(x) for x in tree["right_child"]])
        # tree_str += "\nleaf_value=" + " ".join([str(x) for x in tree["leaf_value"]])
        # tree_str += "\nleaf_weight=" + ("0 " * num_leaves)[:-1]
        # tree_str += "\nleaf_count=" + ("0 " * num_leaves)[:-1]
        # tree_str += "\ninternal_value=" + ("0 " * num_nodes)[:-1]
        # tree_str += "\ninternal_weight=" + ("0 " * num_nodes)[:-1]
        # tree_str += "\ninternal_count=" + ("0 " * num_nodes)[:-1]
        # tree_str += f"\nis_linear={tree['is_linear']}"
        # tree_str += f"\nshrinkage={tree['shrinkage']}"
        # tree_str += "\n\n\n"

        # handle += tree_str

    tree_strings.append(back_str)

    # handle += back_str
    return "".join(tree_strings)
