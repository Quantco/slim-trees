import timeit
import warnings

import pytest

from slim_trees import __version__ as slim_trees_version
from slim_trees import dump_compressed
from slim_trees.pickling import load_compressed


def get_dump_times(model, dump_lib_compressed, tmp_path, method):
    model_path_compressed = tmp_path / "model_compressed"
    model_path = tmp_path / "model"

    dump_time_compressed = timeit.timeit(
        lambda: dump_lib_compressed(model, model_path_compressed, method), number=5
    )
    dump_time_uncompressed = timeit.timeit(
        lambda: dump_compressed(model, model_path, method), number=5
    )
    return dump_time_compressed, dump_time_uncompressed


def get_load_times(model, dump_lib_compressed, tmp_path, method):
    model_path_compressed = tmp_path / "model_compressed"
    model_path = tmp_path / "model"

    dump_lib_compressed(model, model_path_compressed, method)
    dump_compressed(model, model_path, method)
    load_time_compressed = timeit.timeit(
        lambda: load_compressed(model_path_compressed, method), number=5
    )
    load_time_uncompressed = timeit.timeit(
        lambda: load_compressed(model_path, method), number=5
    )
    return load_time_compressed, load_time_uncompressed


def assert_version_pickle(pickle_function, element):
    _, (_, _, (version, _)) = pickle_function(element)
    assert slim_trees_version == version


def assert_version_unpickle(pickle_function, element):
    _unpickle_function, (
        reconstructor,
        args,
        (version, compressed_state),
    ) = pickle_function(element)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _unpickle_function(reconstructor, args, (version, compressed_state))
    with pytest.warns() as record:
        _unpickle_function(reconstructor, args, ("0.0.0", compressed_state))
    assert len(record) == 1
    assert "version mismatch" in str(record[0].message).lower()
