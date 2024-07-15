from typing import Any, Dict

import numpy as np
from numpy.typing import DTypeLike, NDArray

NP_2 = np.__version__ >= "2.0.0"


def can_cast(value: Any, dtype: DTypeLike) -> bool:
    """
    Check if the given value can be cast to the specified data type.

    Notes:
        In numpy 2.0.0, np.can_cast does not support Python scalars anymore and does not apply
          any value-based logic for 0-D arrays and NumPy scalars.
        If value is a uint and dtype is an int of the same size, check if value <= np.iinfo(dtype).max
    """

    if not NP_2:
        return np.can_cast(value, dtype)

    scalar_type = np.min_scalar_type(value)

    return np.can_cast(scalar_type, dtype) or (
        scalar_type.kind == "u"
        and np.issubdtype(dtype, np.signedinteger)
        and value <= np.iinfo(dtype).max
    )


def safe_cast(arr: NDArray, dtype: DTypeLike) -> NDArray:
    if can_cast(arr.max(), dtype) and can_cast(arr.min(), dtype):
        return arr.astype(dtype)
    raise ValueError(f"Cannot cast array to {dtype}.")


def _is_in_neighborhood_of_int(arr: NDArray, iinfo: np.iinfo, eps: float = 1e-12):
    """
    Checks if the numbers are around an integer.
    np.abs(arr % 1 - 1) < eps checks if the number is in an epsilon neighborhood on the right side
    of the next int and arr % 1 < eps checks if the number is in an epsilon neighborhood on the left
    side of the next int.
    """
    return (
        (np.minimum(np.abs(arr % 1 - 1), arr % 1) < eps)
        & (arr >= iinfo.min)
        & (arr <= iinfo.max)
    )


def compress_half_int_float_array(
    a: NDArray, compression_dtype: DTypeLike = "int8"
) -> Dict:
    """Compress small integer and half-integer floats in a lossless fashion

    Idea:
        If most values in array <a> are small integers or half-integers, we can
        store them as float16, while keeping the rest as float64.

    Technical details:
        - The boolean array (2 * a) % 1 == 0 indicates the integers and half-integers in <a>.
        - int8 can represent integers between np.iinfo('int8').min and np.iinfo('int8').max
    """
    info = np.iinfo(compression_dtype)
    a2 = 2.0 * a
    is_compressible = _is_in_neighborhood_of_int(a2, info)

    a2_compressible = a2[is_compressible].astype(compression_dtype)
    a_incompressible = a[~is_compressible]

    state = {
        "is_compressible": np.packbits(is_compressible),
        "a2_compressible": a2_compressible,
        "a_incompressible": a_incompressible,
    }

    return state


def decompress_half_int_float_array(state: Dict) -> NDArray:
    n_thresholds = len(state["a2_compressible"]) + len(state["a_incompressible"])
    is_compressible = np.unpackbits(
        state["is_compressible"], count=n_thresholds
    ).astype("bool")
    a = np.zeros(len(is_compressible), dtype="float64")
    a[is_compressible] = state["a2_compressible"] / 2.0
    a[~is_compressible] = state["a_incompressible"]
    return a
