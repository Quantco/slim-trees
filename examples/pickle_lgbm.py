import pathlib
import tempfile
from typing import Union

import lightgbm as lgb
from lightgbm import Booster
from utils import (
    evaluate_compression_performance,
    evaluate_prediction_difference,
    generate_dataset,
)

from pickle_compression import dump_lgbm_compressed
from pickle_compression.lgbm_booster import dump_lgbm
from pickle_compression.pickling import load_compressed


def train_model() -> lgb.LGBMRegressor:
    regressor = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    X_train, _, y_train, _ = generate_dataset(n_samples=10000)  # noqa: N806
    regressor.fit(X_train, y_train)
    return regressor


def load_model(path) -> Booster:
    return Booster(model_file=path)


def dump_model_string(booster: Booster, path: Union[str, pathlib.Path]):
    with open(path, "w") as f:
        f.write(booster.model_to_string())


if __name__ == "__main__":
    # model = load_model("examples/lgb1-base.model")
    model = train_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "model.pkl"
        dump_lgbm_compressed(model, path, "no")
        model_compressed = load_compressed(path, "no")

    pathlib.Path("examples/out").mkdir(exist_ok=True)
    dump_model_string(model_compressed.booster_, "examples/out/model_compressed.model")

    evaluate_prediction_difference(
        model, model_compressed, generate_dataset(n_samples=10000)[1]
    )
    evaluate_compression_performance(model, dump_lgbm)
