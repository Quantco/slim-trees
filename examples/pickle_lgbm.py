import pathlib
import tempfile
from typing import Union

import lightgbm as lgb
import numpy as np
from lightgbm import Booster
from utils import evaluate_compression_performance, load_data

from pickle_compression import dump_lgbm_compressed
from pickle_compression.lgbm_booster import dump_lgbm
from pickle_compression.pickling import load_compressed


def train_model() -> lgb.LGBMRegressor:
    regressor = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    regressor.fit(*load_data())
    return regressor


def load_model(path) -> Booster:
    return Booster(model_file=path)


def dump_model_string(booster: Booster, path: Union[str, pathlib.Path]):
    with open(path, "w") as f:
        f.write(booster.model_to_string())


# model = load_model("examples/lgb1-base.model")
model = train_model()

with tempfile.TemporaryDirectory() as tmpdir:
    path = pathlib.Path(tmpdir) / "model.pkl"
    dump_lgbm_compressed(model, path, "no")
    model_compressed = load_compressed(path, "no")

pathlib.Path("examples/out").mkdir(exist_ok=True)
dump_model_string(model_compressed.booster_, "examples/out/model_compressed.model")

x, y = load_data()
y_pred = model.predict(x)
y_pred_new = model_compressed.predict(x)
print(f"Maximum prediction difference: {np.abs(y_pred - y_pred_new).max()}")

evaluate_compression_performance(model, dump_lgbm)
