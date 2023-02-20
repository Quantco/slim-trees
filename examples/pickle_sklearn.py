import time
from itertools import product

import pandas as pd
from pickling import get_pickled_size
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn_tree import dump_sklearn


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
    X = df.drop("price", axis=1)  # noqa: N806
    y = df["price"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # noqa: N806
    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X_scaled, y, test_size=0.3, random_state=42
    )

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    return regressor


model = train_model()

for compression, dump_function in product(
    ["no", "lzma", "bz2", "gzip"], [None, dump_sklearn]
):
    start = time.time()
    size = get_pickled_size(model, compression=compression, dump_function=dump_function)
    print(
        f"Compression {compression}, "
        f"dump_function {None if not dump_function else dump_function.__name__}: "
        f"{size / 2**20:.2f} MB / {time.time() - start:.2f} s"
    )
