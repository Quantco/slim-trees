import os
import pickle

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from util import (
    assert_version_pickle,
    assert_version_unpickle,
    get_dump_times,
    get_load_times,
)

from slim_trees import dump_lgbm_compressed, dumps_lgbm_compressed
from slim_trees.lgbm_booster import _booster_pickle
from slim_trees.pickling import dump_compressed, load_compressed, loads_compressed


@pytest.fixture
def lgbm_regressor(rng):
    return LGBMRegressor(random_state=rng)


@pytest.fixture
def lgbm_regressor_linear(rng):
    return LGBMRegressor(random_state=rng, linear_trees=True)


@pytest.mark.parametrize(
    "lgbm_regressor", [lgbm_regressor, lgbm_regressor_linear], indirect=True
)
def test_compressed_predictions(diabetes_toy_df, lgbm_regressor, tmp_path):
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
    assert size_compressed < 0.6 * size


@pytest.mark.xfail(reason="Flaky test")
@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_dump_times(diabetes_toy_df, lgbm_regressor, tmp_path, compression_method):
    lgbm_regressor.fit(*diabetes_toy_df)
    factor = 7 if compression_method == "no" else 4

    dump_time_compressed, dump_time_uncompressed = get_dump_times(
        lgbm_regressor, dump_lgbm_compressed, tmp_path, compression_method
    )
    assert dump_time_compressed < factor * dump_time_uncompressed


@pytest.mark.xfail(reason="Flaky test")
@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_load_times(diabetes_toy_df, lgbm_regressor, tmp_path, compression_method):
    lgbm_regressor.fit(*diabetes_toy_df)

    load_time_compressed, load_time_uncompressed = get_load_times(
        lgbm_regressor, dump_lgbm_compressed, tmp_path, compression_method
    )
    factor = 10 if compression_method == "no" else 7
    assert load_time_compressed < factor * load_time_uncompressed


def test_tree_version_pickle(diabetes_toy_df, lgbm_regressor):
    lgbm_regressor.fit(*diabetes_toy_df)
    assert_version_pickle(_booster_pickle, lgbm_regressor.booster_)


def test_tree_version_unpickle(diabetes_toy_df, lgbm_regressor):
    lgbm_regressor.fit(*diabetes_toy_df)
    assert_version_unpickle(_booster_pickle, lgbm_regressor.booster_)


class _TestUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("lightgbm"):
            raise ImportError(f"Module '{module}' not allowed in this test")
        return super().find_class(module, name)


def test_load_compressed_custom_unpickler(tmp_path, lgbm_regressor):
    model_path = tmp_path / "model_compressed.pickle.lzma"
    dump_lgbm_compressed(lgbm_regressor, model_path)
    with pytest.raises(ImportError, match="lightgbm.*not allowed"):
        load_compressed(model_path, unpickler_class=_TestUnpickler)


def test_loads_compressed_custom_unpickler(lgbm_regressor):
    compressed = dumps_lgbm_compressed(lgbm_regressor)
    with pytest.raises(ImportError, match="lightgbm.*not allowed"):
        loads_compressed(compressed, unpickler_class=_TestUnpickler)


def test_dump_and_load_from_file(tmp_path, lgbm_regressor):
    with (tmp_path / "model.pickle.lzma").open("wb") as file:
        dump_lgbm_compressed(lgbm_regressor, file, compression="lzma")

    with (tmp_path / "model.pickle.lzma").open("rb") as file:
        load_compressed(file, compression="lzma")

    # No compression method specified
    with pytest.raises(ValueError), (tmp_path / "model.pickle.lzma").open("rb") as file:
        load_compressed(file)

    with pytest.raises(ValueError), (tmp_path / "model.pickle.lzma").open("wb") as file:
        dump_lgbm_compressed(lgbm_regressor, file)


# todo add tests for large models
