import pathlib
from typing import Union

import lightgbm as lgb
from lgbm_booster import dump_lgbm
from lightgbm import Booster

from examples.utils import load_data, print_model_size


def train_model() -> lgb.LGBMRegressor:
    regressor = lgb.LGBMRegressor(n_estimators=1, random_state=42)
    regressor.fit(*load_data())
    return regressor


def load_model(path) -> Booster:
    return Booster(model_file=path)


def dump_model_string(booster: Booster, path: Union[str, pathlib.Path]):
    with open(path, "w") as f:
        f.write(booster.model_to_string())


model = train_model()
# dump_model_string(model.booster_, "great_lakes_1.model")

# x, y = load_data()
# model = load_model("great_lakes_1.model")
# model_new = load_model("great_lakes_1_omit_values.model")

# y_pred = model.predict(x)
# y_pred_new = model_new.predict(x)
# diff = y_pred - y_pred_new
# print(diff.max(), diff.min(), diff.mean(), diff.std())

print_model_size(model, dump_lgbm)
