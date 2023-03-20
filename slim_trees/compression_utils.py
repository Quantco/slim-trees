import numpy as np


def safe_cast(arr: np.array, dtype):
    if np.can_cast(arr.max(), dtype) and np.can_cast(arr.min(), dtype):
        return arr.astype(dtype)
    raise ValueError(f"Cannot cast array to {dtype}.")


def _is_in_neighborhood_of_int(arr, iinfo, eps=1e-12):
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


def compress_half_int_float_array(a, compression_dtype="int8"):
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


def decompress_half_int_float_array(state):
    n_thresholds = len(state["a2_compressible"]) + len(state["a_incompressible"])
    is_compressible = np.unpackbits(
        state["is_compressible"], count=n_thresholds
    ).astype("bool")
    a = np.zeros(len(is_compressible), dtype="float64")
    a[is_compressible] = state["a2_compressible"] / 2.0
    a[~is_compressible] = state["a_incompressible"]
    return a
