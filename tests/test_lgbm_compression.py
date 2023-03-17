import os

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from test_util import get_dump_times, get_load_times

from slim_trees import dump_lgbm_compressed
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
    X, y = diabetes_toy_df
    lgbm_regressor.fit(X, y)

    model_path_compressed = tmp_path / "model_compressed.pickle.lzma"
    model_path = tmp_path / "model.pickle.lzma"
    dump_lgbm_compressed(lgbm_regressor, model_path_compressed)
    dump_compressed(lgbm_regressor, model_path)
    size_compressed = os.path.getsize(model_path_compressed)
    size = os.path.getsize(model_path)
    assert size_compressed < 0.7 * size


@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_dump_times(diabetes_toy_df, lgbm_regressor, tmp_path, compression_method):
    X, y = diabetes_toy_df
    lgbm_regressor.fit(X, y)
    factor = 22 if compression_method == "no" else 10

    dump_time_compressed, dump_time_uncompressed = get_dump_times(
        lgbm_regressor, dump_lgbm_compressed, tmp_path, compression_method
    )

    # compressed should only take [factor] times longer than uncompressed
    assert dump_time_compressed / dump_time_uncompressed < factor, \
        f"factor for compressed too high: {dump_time_compressed / dump_time_uncompressed:.4f} > {factor}"


@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_load_times(diabetes_toy_df, lgbm_regressor, tmp_path, compression_method):
    X, y = diabetes_toy_df
    lgbm_regressor.fit(X, y)

    load_time_compressed, load_time_uncompressed = get_load_times(
        lgbm_regressor, dump_lgbm_compressed, tmp_path, compression_method
    )
    factor = 35 if compression_method == "no" else 15
    # compressed should only take [factor] times longer than uncompressed
    assert load_time_compressed / load_time_uncompressed < factor, \
        f"{load_time_compressed / load_time_uncompressed:.2f} > {factor}"

# todo add tests for large models
