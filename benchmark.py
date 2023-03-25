import io
import lzma
import pickle
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, List

import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from examples.utils import generate_dataset
from slim_trees.lgbm_booster import dump_lgbm
from slim_trees.sklearn_tree import dump_sklearn

MODELS_PATH = "examples/benchmark_models"

def onnx_stuff(model):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([None, 100]))]
    print("Converting sklearn")
    onx = convert_sklearn(model, initial_types=initial_type)
    print("Done")
    # onx_string = onx.SerializeToString()
    with open("examples/benchmark_models/rf_sklearn_large.onnx", "wb") as f:
        print("Writing")
        f.write(onx.SerializeToString())
        # onx.SerializeToString()
        print("Done")
        # print(len(onx_string))



def load_model(model_name: str, generate: Callable) -> Any:
    model_path = Path(f"{MODELS_PATH}/{model_name}.pkl")

    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)

    regressor = generate()
    regressor.fit(*generate_dataset(n_samples=10000))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(regressor, f)
    return regressor


def train_gb_sklearn() -> GradientBoostingRegressor:
    return load_model(
        "gb_sklearn",
        lambda: GradientBoostingRegressor(n_estimators=2000, random_state=42),
    )


def train_large_tree_sklearn() -> RandomForestRegressor:
    return load_model(
        "rf_sklearn_large",
        lambda: RandomForestRegressor(
            n_estimators=1500, max_leaf_nodes=10000, random_state=42, n_jobs=-1
        ),
    )


def train_model_sklearn() -> RandomForestRegressor:
    return load_model(
        "rf_sklearn",
        lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    )


def train_gbdt_lgbm() -> lgb.LGBMRegressor:
    return load_model(
        "gbdt_lgbm", lambda: lgb.LGBMRegressor(n_estimators=2000, random_state=42)
    )


def train_gbdt_large_lgbm() -> lgb.LGBMRegressor:
    return load_model(
        "gbdt_large_lgbm",
        lambda: lgb.LGBMRegressor(n_estimators=20000, random_state=42),
    )


def train_rf_lgbm() -> lgb.LGBMRegressor:
    return load_model(
        "rg_lgbm",
        lambda: lgb.LGBMRegressor(
            boosting_type="rf",
            n_estimators=100,
            num_leaves=1000,
            random_state=42,
            bagging_freq=5,
            bagging_fraction=0.5,
            verbose=-1,
        ),
    )


def benchmark(func: Callable, *args, **kwargs) -> float:
    start = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - start


def benchmark_model(  # noqa: PLR0913
    name,
    train_func,
    dumps_func,
    loads_func=None,
    base_dumps_func=None,
    base_loads_func=None,
) -> dict:
    if loads_func is None:
        loads_func = pickle.loads
    if base_dumps_func is None:
        base_dumps_func = pickle.dumps
    if base_loads_func is None:
        base_loads_func = pickle.loads

    model = train_func()
    onnx_stuff(model)
    exit(0)

    naive_dump_time = benchmark(base_dumps_func, model)
    naive_pickled = base_dumps_func(model)
    naive_pickled_size = len(naive_pickled)
    naive_load_time = benchmark(base_loads_func, naive_pickled)

    our_dump_time = benchmark(dumps_func, model)
    our_pickled = dumps_func(model)
    our_pickled_size = len(our_pickled)
    our_load_time = benchmark(loads_func, our_pickled)
    return {
        "name": name,
        "baseline": {
            "size": naive_pickled_size,
            "dump_time": naive_dump_time,
            "load_time": naive_load_time,
        },
        "ours": {
            "size": our_pickled_size,
            "dump_time": our_dump_time,
            "load_time": our_load_time,
        },
        "change": {
            "size": naive_pickled_size / our_pickled_size,
            "dump_time": our_dump_time / naive_dump_time,
            "load_time": our_load_time / naive_load_time,
        },
    }


def format_size(n_bytes: int) -> str:
    MiB = 1024**2
    return f"{n_bytes / MiB:.1f} MiB"


def format_time(seconds: float) -> str:
    return f"{seconds:.2f} s"


def format_change(multiple: float) -> str:
    return f"{multiple:.2f} x"


def format_benchmarks_results_table(benchmark_results: List[dict]) -> str:
    header = """
        | Model | Size | Dump Time | Load Time |
        |--|--:|--:|--:|
    """

    def format_row(results):
        def format_cell(base, ours, change):
            return f"{base} / {ours} / {change}"

        column_data = [
            results["name"],
            format_cell(
                format_size(results["baseline"]["size"]),
                format_size(results["ours"]["size"]),
                format_change(results["change"]["size"]),
            ),
            format_cell(
                format_time(results["baseline"]["dump_time"]),
                format_time(results["ours"]["dump_time"]),
                format_change(results["change"]["dump_time"]),
            ),
            format_cell(
                format_time(results["baseline"]["load_time"]),
                format_time(results["ours"]["load_time"]),
                format_change(results["change"]["load_time"]),
            ),
        ]
        return " | ".join(column_data)

    formatted_rows = map(format_row, benchmark_results)

    return (textwrap.dedent(header) + "\n".join(formatted_rows)).strip()


def dumps(model, dump_func):
    bytes_buf = io.BytesIO()
    dump_func(model, bytes_buf)
    return bytes_buf.getvalue()


def dumps_lzma(model, dump_func=None):
    if dump_func is None:
        dump_func = pickle.dump
    bytes_buf = io.BytesIO()
    dump_func(model, bytes_buf)
    return lzma.compress(bytes_buf.getvalue())


def loads_lzma(data):
    decompressed = lzma.decompress(data)
    return pickle.loads(decompressed)


if __name__ == "__main__":
    dumps_sklearn_args = (lambda x: dumps(x, dump_sklearn),)
    dumps_lgbm_args = (lambda x: dumps(x, dump_lgbm),)
    dumps_sklearn_lzma_args = (
        lambda x: dumps_lzma(x, dump_sklearn),
        loads_lzma,
        dumps_lzma,
        loads_lzma,
    )
    dumps_lgbm_lzma_args = (
        lambda x: dumps_lzma(x, dump_lgbm),
        loads_lzma,
        dumps_lzma,
        loads_lzma,
    )
    models_to_benchmark = [
        # ("sklearn rf", train_model_sklearn) + dumps_sklearn_args,
        # ("sklearn rf LZMA", train_model_sklearn) + dumps_sklearn_lzma_args,
        ("sklearn rf large", train_large_tree_sklearn) + dumps_sklearn_args,
        # ("sklearn rf large LZMA", train_large_tree_sklearn) + dumps_sklearn_lzma_args,
        # ("sklearn gb", train_gb_sklearn) + dumps_sklearn_args,
        # ("sklearn gb LZMA", train_gb_sklearn) + dumps_sklearn_lzma_args,
        # ("LGBM gbdt", train_gbdt_lgbm) + dumps_lgbm_args,
        # ("LGBM gbdt LZMA", train_gbdt_lgbm) + dumps_lgbm_lzma_args,
        # ("LGBM gbdt large", train_gbdt_large_lgbm) + dumps_lgbm_args,
        # ("LGBM gbdt large LZMA", train_gbdt_large_lgbm) + dumps_lgbm_lzma_args,
        # ("LGBM rf", train_rf_lgbm) + dumps_lgbm_args,
        # ("LGBM rf LZMA", train_rf_lgbm) + dumps_lgbm_lzma_args,
    ]
    benchmark_results = [benchmark_model(*args) for args in models_to_benchmark]
    print("Base results / Our results / Change")
    print(format_benchmarks_results_table(benchmark_results))
