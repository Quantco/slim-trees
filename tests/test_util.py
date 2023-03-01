import timeit

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
