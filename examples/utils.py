import os
import tempfile
import time
from itertools import product
from typing import Any, Callable, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from pickle_compression.pickling import dump_compressed, load_compressed


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("examples/great_lakes_1.csv")
    df.drop(["lat", "long"], axis=1, inplace=True)
    cols = ["region", "type", "laundry_options", "parking_options"]
    label_encoder = LabelEncoder()
    mapping_dict = {}
    for col in cols:
        df[col] = label_encoder.fit_transform(df[col])
        le_name_mapping = dict(
            zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        )
        mapping_dict[col] = le_name_mapping
    X = df.drop("price", axis=1)  # noqa: N806
    y = df["price"]
    return X, y


def evaluate_compression_performance(
    model: Any, dump: Callable, print_performance: bool = True
):
    compressions = ["no", "lzma", "bz2", "gzip"]
    performance = []
    for compression, dump_function in product(compressions, [None, dump]):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model"
            start = time.time()
            dump_compressed(model, path, compression, dump_function)
            dump_time = time.time() - start
            start = time.time()
            load_compressed(path, compression)
            load_time = time.time() - start
            size = os.path.getsize(path)
        performance += [
            {
                "compression": compression,
                "dump_function": dump_function.__name__ if dump_function else None,
                "size": f"{size / 2 ** 20:.2f} MB",
                "dump_time": f"{dump_time:.3f} s",
                "load_time": f"{load_time:.3f} s",
            }
        ]
    df = pd.DataFrame(performance)
    if print_performance:
        print(df.to_string(index=False))
    return df
