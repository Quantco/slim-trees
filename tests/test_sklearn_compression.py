import os
import pickle

import numpy as np
import pytest
from packaging.version import Version
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from util import (
    assert_version_pickle,
    assert_version_unpickle,
    get_dump_times,
    get_load_times,
)

from slim_trees import dump_sklearn_compressed, dumps_sklearn_compressed
from slim_trees.pickling import dump_compressed, load_compressed, loads_compressed
from slim_trees.sklearn_tree import _tree_pickle


@pytest.fixture
def random_forest_regressor(rng):
    return RandomForestRegressor(n_estimators=100, random_state=rng)


@pytest.fixture
def random_forest_regressor_large(rng):
    return RandomForestRegressor(n_estimators=1000, random_state=rng)


@pytest.fixture
def decision_tree_regressor(rng):
    return DecisionTreeRegressor(random_state=rng)


def test_compressed_predictions(diabetes_toy_df, random_forest_regressor, tmp_path):
    X, y = diabetes_toy_df
    random_forest_regressor.fit(*diabetes_toy_df)

    model_path = tmp_path / "model_compressed.pickle.lzma"
    dump_sklearn_compressed(random_forest_regressor, model_path)
    model_dtype_reduction = load_compressed(model_path, "lzma")
    prediction_no_reduction = random_forest_regressor.predict(X)
    prediction_reduction = model_dtype_reduction.predict(X)
    np.testing.assert_allclose(prediction_no_reduction, prediction_reduction)


def test_compressed_internal_structure(
    diabetes_toy_df, decision_tree_regressor, tmp_path
):
    X, y = diabetes_toy_df
    decision_tree_regressor.fit(X, y)

    model_path = tmp_path / "model_dtype_reduction.pickle.lzma"
    dump_sklearn_compressed(decision_tree_regressor, model_path)
    model_dtype_reduction = load_compressed(model_path, "lzma")

    tree_no_reduction = decision_tree_regressor.tree_
    tree_dtype_reduction = model_dtype_reduction.tree_

    np.testing.assert_array_equal(
        tree_dtype_reduction.children_left, tree_no_reduction.children_left
    )
    np.testing.assert_array_equal(
        tree_dtype_reduction.children_right, tree_no_reduction.children_right
    )
    np.testing.assert_array_equal(
        tree_dtype_reduction.feature, tree_no_reduction.feature
    )
    # threshold compression should be lossless (even for float compression)
    np.testing.assert_array_equal(
        tree_dtype_reduction.threshold, tree_no_reduction.threshold
    )
    is_leaf = tree_dtype_reduction.children_left == -1
    np.testing.assert_allclose(
        tree_dtype_reduction.value[is_leaf], tree_no_reduction.value[is_leaf]
    )
    from sklearn import __version__ as _sklearn_version

    sklearn_version = Version(_sklearn_version)
    sklearn_version_ge_130 = sklearn_version >= Version("1.3")
    if sklearn_version_ge_130:
        np.testing.assert_allclose(
            tree_dtype_reduction.missing_go_to_left[~is_leaf],
            tree_no_reduction.missing_go_to_left[~is_leaf],
        )


def test_compression_size(diabetes_toy_df, random_forest_regressor, tmp_path):
    X, y = diabetes_toy_df
    random_forest_regressor.fit(X, y)

    model_path_dtype_reduction = tmp_path / "model_dtype_reduction.pickle.lzma"
    model_path_no_reduction = tmp_path / "model_no_reduction.pickle.lzma"
    dump_sklearn_compressed(random_forest_regressor, model_path_dtype_reduction)
    dump_compressed(random_forest_regressor, model_path_no_reduction)
    size_no_reduction = os.path.getsize(model_path_no_reduction)
    size_dtype_reduction = os.path.getsize(model_path_dtype_reduction)
    assert size_dtype_reduction < 0.5 * size_no_reduction


@pytest.mark.xfail(reason="Flaky test")
@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_dump_times(
    diabetes_toy_df, random_forest_regressor, tmp_path, compression_method
):
    X, y = diabetes_toy_df
    random_forest_regressor.fit(X, y)
    factor = 4 if compression_method == "no" else 1.5

    dump_time_compressed, dump_time_uncompressed = get_dump_times(
        random_forest_regressor, dump_sklearn_compressed, tmp_path, compression_method
    )
    assert dump_time_compressed < factor * dump_time_uncompressed


@pytest.mark.xfail(reason="Flaky test")
@pytest.mark.parametrize("compression_method", ["no", "lzma", "gzip", "bz2"])
def test_load_times(
    diabetes_toy_df, random_forest_regressor, tmp_path, compression_method
):
    X, y = diabetes_toy_df
    random_forest_regressor.fit(X, y)

    load_time_compressed, load_time_uncompressed = get_load_times(
        random_forest_regressor, dump_sklearn_compressed, tmp_path, compression_method
    )
    factor = 8 if compression_method == "no" else 1.5
    assert load_time_compressed < factor * load_time_uncompressed


def test_tree_version_pickle(diabetes_toy_df, decision_tree_regressor):
    decision_tree_regressor.fit(*diabetes_toy_df)
    assert_version_pickle(_tree_pickle, decision_tree_regressor.tree_)


def test_tree_version_unpickle(diabetes_toy_df, decision_tree_regressor):
    decision_tree_regressor.fit(*diabetes_toy_df)
    assert_version_unpickle(_tree_pickle, decision_tree_regressor.tree_)


class _TestUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("sklearn"):
            raise ImportError(f"Module '{module}' not allowed in this test")
        return super().find_class(module, name)


def test_load_compressed_custom_unpickler(tmp_path, random_forest_regressor):
    model_path = tmp_path / "model_compressed.pickle.lzma"
    dump_sklearn_compressed(random_forest_regressor, model_path)
    with pytest.raises(ImportError, match="sklearn.*not allowed"):
        load_compressed(model_path, unpickler_class=_TestUnpickler)


def test_loads_compressed_custom_unpickler(random_forest_regressor):
    compressed = dumps_sklearn_compressed(random_forest_regressor)
    with pytest.raises(ImportError, match="sklearn.*not allowed"):
        loads_compressed(compressed, unpickler_class=_TestUnpickler)


def test_dump_and_load_from_file(tmp_path, random_forest_regressor):
    with (tmp_path / "model.pickle.lzma").open("wb") as file:
        dump_sklearn_compressed(random_forest_regressor, file, compression="lzma")

    with (tmp_path / "model.pickle.lzma").open("rb") as file:
        load_compressed(file, compression="lzma")

    # No compression method specified
    with pytest.raises(ValueError), (tmp_path / "model.pickle.lzma").open("rb") as file:
        load_compressed(file)

    with pytest.raises(ValueError), (tmp_path / "model.pickle.lzma").open("wb") as file:
        dump_sklearn_compressed(random_forest_regressor, file)


def test_too_many_leaves_for_unit16(tmp_path):
    x = np.arange(70_000).reshape(-1, 1)
    y = np.arange(70_000)

    # model = RandomForestRegressor(max_depth=1, n_estimators=70_000)
    model = DecisionTreeRegressor(random_state=42)
    model.fit(x, y)

    dump_sklearn_compressed(model, tmp_path / "model.pickle.lzma")

    with (tmp_path / "model.pickle.lzma").open("rb") as file:
        model_loaded = load_compressed(file, compression="lzma")

    assert model.predict([[1234]]) == model_loaded.predict([[1234]])


# todo add tests for large models
