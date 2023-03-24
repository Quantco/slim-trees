import os

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from util import (
    assert_version_pickle,
    assert_version_unpickle,
    get_dump_times,
    get_load_times,
)

from slim_trees import dump_lgbm_compressed
from slim_trees.lgbm_booster import _booster_pickle
from slim_trees.pickling import dump_compressed, load_compressed


@pytest.fixture(params=[False, True])
def lgbm_regressor(rng, request):
    return LGBMRegressor(random_state=rng, linear_trees=request.param)


def test_compresed_predictions(diabetes_toy_df, lgbm_regressor, tmp_path):
    X, y = diabetes_toy_df
    lgbm_regressor.fit(X, y)

    model_path = tmp_path / "model_compressed.pickle.lzma"
    dump_lgbm_compressed(lgbm_regressor, model_path)
    model_compressed = load_compressed(model_path, "lzma")
    prediction = lgbm_regressor.predict(X)
    prediction_compressed = model_compressed.predict(X)
    np.testing.assert_allclose(prediction, prediction_compressed)


def test_compressed_size(diabetes_toy_df, lgbm_regressor, tmp_path):
    lgbm_regressor.fit(*diabetes_toy_df)

    model_path_compressed = tmp_path / "model_compressed.pickle.lzma"
    model_path = tmp_path / "model.pickle.lzma"
    dump_lgbm_compressed(lgbm_regressor, model_path_compressed)
    dump_compressed(lgbm_regressor, model_path)
    size_compressed = os.path.getsize(model_path_compressed)
    size = os.path.getsize(model_path)
    factor = 0.85 if lgbm_regressor.linear_trees else 0.7
    assert size_compressed < factor * size


@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_dump_times(diabetes_toy_df, lgbm_regressor, tmp_path, compression_method):
    lgbm_regressor.fit(*diabetes_toy_df)
    factor = 22 if compression_method == "no" else 10

    time_compressed, time_uncompressed = get_dump_times(
        lgbm_regressor, dump_lgbm_compressed, tmp_path, compression_method
    )

    # compressed should only take [factor] times longer than uncompressed
    assert (
        time_compressed / time_uncompressed < factor
    ), f"factor for compressed too high: {time_compressed / time_uncompressed:.4f} > {factor}"


@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_load_times(diabetes_toy_df, lgbm_regressor, tmp_path, compression_method):
    lgbm_regressor.fit(*diabetes_toy_df)

    time_compressed, time_uncompressed = get_load_times(
        lgbm_regressor, dump_lgbm_compressed, tmp_path, compression_method
    )
    factor = 35 if compression_method == "no" else 15
    # compressed should only take [factor] times longer than uncompressed
    assert (
        time_compressed / time_uncompressed < factor
    ), f"{time_compressed / time_uncompressed:.2f} > {factor}"


def test_tree_version_pickle(diabetes_toy_df, lgbm_regressor):
    lgbm_regressor.fit(*diabetes_toy_df)
    assert_version_pickle(_booster_pickle, lgbm_regressor.booster_)


def test_tree_version_unpickle(diabetes_toy_df, lgbm_regressor):
    lgbm_regressor.fit(*diabetes_toy_df)
    assert_version_unpickle(_booster_pickle, lgbm_regressor.booster_)


# todo add tests for large models
