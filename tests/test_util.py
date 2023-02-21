import timeit

from pickling import load_compressed

from pickle_compression import dump_compressed


def get_compression_times(model, dump_lib_compressed, tmp_path, method):
    model_path_compressed = tmp_path / "model_compressed"
    model_path = tmp_path / "model"

    dump_time_compressed = timeit.timeit(
        lambda: dump_lib_compressed(model, model_path_compressed, method), number=5
    )
    dump_time_uncompressed = timeit.timeit(
        lambda: dump_compressed(model, model_path, method), number=5
    )
    load_time_compressed = timeit.timeit(
        lambda: load_compressed(model_path_compressed, method), number=5
    )
    load_time_uncompressed = timeit.timeit(
        lambda: load_compressed(model_path, method), number=5
    )
    return (
        dump_time_compressed,
        load_time_compressed,
        dump_time_uncompressed,
        load_time_uncompressed,
    )
