import os

import numpy as np
import pytest as pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.tests.test_bagging import diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

from pickle_compression.compress_model import dump_compressed_dtype_reduction, compress_half_int_float_array, \
    decompress_half_int_float_array, _is_in_neighborhood_of_int
from pickle_compression.pickling import load_compressed, dump_compressed


@pytest.fixture
def diabetes_toy_df():
    return diabetes.data[:50], diabetes.target[:50]


@pytest.fixture
def rng():
    return check_random_state(0)


@pytest.fixture
def random_forest_regressor(rng):
    return RandomForestRegressor(n_estimators=100, random_state=rng)


@pytest.fixture
def decision_tree_regressor(rng):
    return DecisionTreeRegressor(random_state=rng)


def test_compressed_predictions(diabetes_toy_df, random_forest_regressor, tmp_path):
    X, y = diabetes_toy_df
    random_forest_regressor.fit(X, y)

    model_path = tmp_path / 'model_dtype_reduction.pickle.lzma'
    dump_compressed_dtype_reduction(random_forest_regressor, model_path)
    model_dtype_reduction = load_compressed(model_path, 'lzma')
    prediction_no_reduction = random_forest_regressor.predict(X)
    prediction_reduction = model_dtype_reduction.predict(X)
    np.testing.assert_allclose(prediction_no_reduction, prediction_reduction)


def test_compressed_internal_structure(diabetes_toy_df, decision_tree_regressor, tmp_path):
    X, y = diabetes_toy_df
    decision_tree_regressor.fit(X, y)

    model_path = tmp_path / 'model_dtype_reduction.pickle.lzma'
    dump_compressed_dtype_reduction(decision_tree_regressor, model_path)
    model_dtype_reduction = load_compressed(model_path, 'lzma')

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


def test_compression_size(diabetes_toy_df, random_forest_regressor, tmp_path):
    X, y = diabetes_toy_df
    random_forest_regressor.fit(X, y)

    model_path_dtype_reduction = tmp_path / 'model_dtype_reduction.pickle.lzma'
    model_path_no_reduction = tmp_path / 'model_no_reduction.pickle.lzma'
    dump_compressed_dtype_reduction(random_forest_regressor, model_path_dtype_reduction)
    dump_compressed(random_forest_regressor, model_path_no_reduction)
    size_no_reduction = os.path.getsize(model_path_no_reduction)
    size_dtype_reduction = os.path.getsize(model_path_dtype_reduction)
    assert size_dtype_reduction < .5 * size_no_reduction


def test_compress_half_int_float_array():
    a1 = np.array([0, 1, 2.5, np.pi, -np.pi, 1e5, 35.5, 2.50000000001])
    state = compress_half_int_float_array(a1)
    np.testing.assert_array_equal(a1, decompress_half_int_float_array(state))


def test_compress_is_compressible_edge_cases():
    a2 = np.array([1.9999999999999, 2.0000000000001])
    is_compressible = _is_in_neighborhood_of_int(a2, np.iinfo('int8'), eps=1e-12)
    assert np.all(is_compressible)
