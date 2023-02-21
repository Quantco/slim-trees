import numpy as np
import pytest
from sklearn.ensemble.tests.test_bagging import diabetes


@pytest.fixture()
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def diabetes_toy_df():
    return diabetes.data, diabetes.target
