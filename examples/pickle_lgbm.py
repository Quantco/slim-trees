import time
from itertools import product

import lightgbm as lgb
import pandas as pd
from lgbm_booster import dump_lgbm
from lightgbm import Booster
from pickling import get_pickled_size
from sklearn.preprocessing import LabelEncoder


def train_model():
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
    regressor = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    X = df.drop("price", axis=1)  # noqa: N806
    y = df["price"]
    regressor.fit(X, y)
    return regressor


def load_model():
    return Booster(model_file="lgb1.model")


model = load_model()

for compression, dump_function in product(
    ["no", "lzma", "bz2", "gzip"], [None, dump_lgbm]
):
    start = time.time()
    size = get_pickled_size(model, compression=compression, dump_function=dump_function)
    print(
        f"Compression {compression}, "
        f"dump_function {None if not dump_function else dump_function.__name__}: "
        f"{size / 2**20:.2f} MB / {time.time() - start:.2f} s"
    )
