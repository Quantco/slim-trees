import time
from itertools import product

import pandas as pd
from pickling import get_pickled_size
from sklearn.preprocessing import LabelEncoder


def load_data():
    df = pd.read_csv("great_lakes_1.csv")
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


def print_model_size(model, dump, compressions=None):
    if not compressions:
        compressions = ["no", "lzma", "bz2", "gzip"]
    for compression, dump_function in product(compressions, [None, dump]):
        start = time.time()
        size = get_pickled_size(
            model, compression=compression, dump_function=dump_function
        )
        print(
            f"Compression {compression}, "
            f"dump_function {None if not dump_function else dump_function.__name__}: "
            f"{size / 2 ** 20:.2f} MB / {time.time() - start:.2f} s"
        )
