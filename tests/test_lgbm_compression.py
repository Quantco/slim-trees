import os

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from test_util import get_dump_times, get_load_times

from pickle_compression import dump_lgbm_compressed
from pickle_compression.pickling import dump_compressed, load_compressed


@pytest.fixture
def lgbm_regressor(rng):
    return LGBMRegressor(random_state=rng)


def test_compresed_predictions(diabetes_toy_df, lgbm_regressor, tmp_path):
    X, y = diabetes_toy_df  # noqa: N806
    lgbm_regressor.fit(X, y)

    model_path = tmp_path / "model_compressed.pickle.lzma"
    dump_lgbm_compressed(lgbm_regressor, model_path)
    model_compressed = load_compressed(model_path, "lzma")
    prediction = lgbm_regressor.predict(X)
    prediction_compressed = model_compressed.predict(X)
    np.testing.assert_allclose(prediction, prediction_compressed)


def test_compressed_size(diabetes_toy_df, lgbm_regressor, tmp_path):
    X, y = diabetes_toy_df  # noqa: N806
    lgbm_regressor.fit(X, y)

    model_path_compressed = tmp_path / "model_compressed.pickle.lzma"
    model_path = tmp_path / "model.pickle.lzma"
    dump_lgbm_compressed(lgbm_regressor, model_path_compressed)
    dump_compressed(lgbm_regressor, model_path)
    size_compressed = os.path.getsize(model_path_compressed)
    size = os.path.getsize(model_path)
    assert size_compressed < 0.6 * size


@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_dump_times(diabetes_toy_df, lgbm_regressor, tmp_path, compression_method):
    X, y = diabetes_toy_df  # noqa: N806
    lgbm_regressor.fit(X, y)
    factor = 7 if compression_method == "no" else 4

    dump_time_compressed, dump_time_uncompressed = get_dump_times(
        lgbm_regressor, dump_lgbm_compressed, tmp_path, compression_method
    )
    assert dump_time_compressed < factor * dump_time_uncompressed


@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_load_times(diabetes_toy_df, lgbm_regressor, tmp_path, compression_method):
    X, y = diabetes_toy_df  # noqa: N806
    lgbm_regressor.fit(X, y)

    load_time_compressed, load_time_uncompressed = get_load_times(
        lgbm_regressor, dump_lgbm_compressed, tmp_path, compression_method
    )
    factor = 10 if compression_method == "no" else 7
    assert load_time_compressed < factor * load_time_uncompressed
