import numpy as np
import pytest

from slim_trees.compression_utils import (
    _is_in_neighborhood_of_int,
    compress_half_int_float_array,
    decompress_half_int_float_array,
    safe_cast,
)


@pytest.mark.parametrize(
    "arr,dtype",
    [
        (np.array([1, 2, 3, 4, 5]), np.int8),
        (np.array([1, 2, 3, 4, 5]), np.uint8),
        (np.array([1, 2, 3, 4, 5]), np.int16),
        (np.array([200]), np.uint16),
    ],
)
def test_safe_cast(arr, dtype):
    safe_cast(arr, dtype)


@pytest.mark.parametrize(
    "arr,dtype",
    [
        (np.array([1, 2, 3, 555555]), np.int16),
        (np.array([-1, 4, 6]), np.uint32),
    ],
)
def test_safe_cast_error(arr, dtype):
    with pytest.raises(ValueError):
        safe_cast(arr, dtype)


def test_compress_half_int_float_array():
    a1 = np.array([0, 1, 2.5, np.pi, -np.pi, 1e5, 35.5, 2.50000000001])
    state = compress_half_int_float_array(a1)
    np.testing.assert_array_equal(a1, decompress_half_int_float_array(state))


def test_compress_is_compressible_edge_cases():
    a2 = np.array([1.9999999999999, 2.0000000000001])
    is_compressible = _is_in_neighborhood_of_int(a2, np.iinfo("int8"), eps=1e-12)
    assert np.all(is_compressible)
