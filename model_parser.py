import pickle
from curses.ascii import isdigit
import re
from typing import Union
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

TREE_GROUP_REGEX = r"(Tree=\d+\n+)((?:.+\n)*)\n\n"
TREE_FEATURES = ["is_linear", "shrinkage"]


def _extract_feature(feature_line):
    feat_name, values_str = feature_line.split("=")
    return feat_name, values_str.split(" ")


def _transform_value(a: str) -> Union[int, float]:
    if a.isdigit():
        return int(a)
    elif a.replace(".", "", 1).isdigit() and a.count(".") < 2:
        return float(a)
    else:
        return a


def df_from_model_string(model_str: str, transform_values=False) -> pd.DataFrame:
    res: list[dict] = []
    tree_matches = re.findall(TREE_GROUP_REGEX, model_str)
    for tree_match in tree_matches:
        tree_name, features_list = tree_match
        _, tree_idx = tree_name.replace("\n", "").split("=")

        # extract features -- filter out empty ones
        features = [f for f in features_list.split("\n") if "=" in f]

        # get number of leaves in tree
        _, num_leaves = _extract_feature(features.pop(0))
        _, num_cat = _extract_feature(features.pop(0))  # unnecessary

        feats_map = dict(_extract_feature(fl) for fl in features)
        for node_idx in range(int(num_leaves[0]) - 1):
            res.append(
                {
                    "tree_idx": int(tree_idx),
                    "node_idx": f"{tree_idx}-{node_idx}",
                    **{
                        feat_name: values[node_idx]
                        if not transform_values
                        else _transform_value(values[node_idx])
                        for feat_name, values in feats_map.items()
                        if feat_name not in TREE_FEATURES
                    },  # all node specific features
                }
            )

    return pd.DataFrame(res)


def get_type(s: str):
    if s.isdigit():
        return int
    elif s.isdecimal():
        return float
    else:
        return str

def pyarrow_table_to_bytes(table: pa.Table) -> bytes:
    stream = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(stream, table.schema)
    writer.write_table(table)
    writer.close()
    return stream.getvalue().to_pybytes()


if __name__ == "__main__":
    # df = df_from_model_string(open("test_model_string.model", "r").read())
    df = df_from_model_string(open("private_/lgb1.txt", "r").read())
    # TODO: cannot be interpreted directly by parquet (pyarrow & fastparquet) as of now.
    # dfo = df_from_model_string(open("lgb1.txt", "r").read(), transform_values=True)
    print(df.head(50))

    df = df.convert_dtypes(infer_objects=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    df.to_parquet("private_/lgb1_conv.parquet")
    table: pa.Table = pa.Table.from_pandas(df).drop(
        [
            "node_idx",
            "split_gain",
            "leaf_weight",
            "leaf_count",
            "internal_value",
            "internal_weight",
            "internal_count",
        ]
    )
    target_schema = pa.schema(
        [
            pa.field("tree_idx", pa.int16()),
            pa.field("split_feature", pa.int16()),
            # pa.field('split_gain', pa.float64()),
            pa.field(
                "threshold", pa.float64()
            ),  # TODO: half int array has the same size effect, validate.
            pa.field("decision_type", pa.int8()),
            pa.field("left_child", pa.int8()),
            pa.field("right_child", pa.int8()),
            pa.field("leaf_value", pa.float64()),
            # pa.field('leaf_weight', pa.float64()),
            # pa.field('leaf_count', pa.int32()),
            # pa.field('internal_value', pa.float64()),
            # pa.field('internal_weight', pa.float64()),
            # pa.field('internal_count', pa.int32()),
        ]
    )
    table_enc = pa.Table.from_arrays([*table], schema=target_schema)
    pq.write_table(table_enc, "private_/lgb1.parquet.lz4", compression="lz4")
    table_bytes = pyarrow_table_to_bytes(table_enc)
    pickle.dump(table_bytes, open("private_/byte_dumps/lgb1.parquet.pkl", "wb"))

    # recreate again
    table_bytes_loaded = pickle.load(open("private_/byte_dumps/lgb1.parquet.pkl", "rb"))
    table_loaded = pa.ipc.open_stream(table_bytes_loaded).read_all()

    # check if tables are equal
    assert table_enc.equals(table_loaded)
    exit(0)
