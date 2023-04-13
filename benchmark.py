import io
import itertools
import lzma
import pickle
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, List

import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from examples.utils import generate_dataset
from slim_trees.lgbm_booster import dump as dump_lgbm
from slim_trees.pickling import get_pickled_size
from slim_trees.sklearn_tree import dump as dump_sklearn

MODELS_PATH = "examples/benchmark_models"


def load_model(model_name: str, generate: Callable) -> Any:
    model_path = Path(f"{MODELS_PATH}/{model_name}.pkl")

    if model_path.exists():
        print(f"Loading model `{model_name}.pkl` from disk...")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    print(f"Training model `{model_name}`...")
    regressor = generate()
    regressor.fit(
        *generate_dataset(n_samples=10000),
    )
    size = get_pickled_size(regressor, "no", pickle.dump)
    print(f"Trained model {model_name}. Size {size / 2**20:.2f} MB")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(regressor, f)
    return regressor


def train_sklearn_rf_20m() -> RandomForestRegressor:
    return load_model(
        "sklearn_rf_20m",
        lambda: RandomForestRegressor(
            n_estimators=100, max_leaf_nodes=1700, random_state=42, n_jobs=-1
        ),
    )


def train_sklearn_rf_200m() -> RandomForestRegressor:
    return load_model(
        "sklearn_rf_200m",
        lambda: RandomForestRegressor(n_estimators=275, random_state=42, n_jobs=-1),
    )


def train_sklearn_rf_1g() -> RandomForestRegressor:
    return load_model(
        "sklearn_rf_1g",
        lambda: RandomForestRegressor(
            n_estimators=1500, max_leaf_nodes=10000, random_state=42, n_jobs=-1
        ),
    )


def train_sklearn_gb_2m() -> GradientBoostingRegressor:
    return load_model(
        "sklearn_gb_2m",
        lambda: GradientBoostingRegressor(
            n_estimators=2000, random_state=42, verbose=True
        ),
    )


def train_lgbm_gbdt_2m() -> lgb.LGBMRegressor:
    return load_model(
        "lgbm_gbdt_2m", lambda: lgb.LGBMRegressor(n_estimators=1000, random_state=42)
    )


def train_lgbm_gbdt_5m() -> lgb.LGBMRegressor:
    return load_model(
        "lgbm_gbdt_5m",
        lambda: lgb.LGBMRegressor(n_estimators=2000, random_state=42),
    )


def train_lgbm_gbdt_20m() -> lgb.LGBMRegressor:
    return load_model(
        "lgbm_gbdt_20m",
        lambda: lgb.LGBMRegressor(n_estimators=8000, random_state=42),
    )


def train_lgbm_gbdt_100m() -> lgb.LGBMRegressor:
    return load_model(
        "lgbm_gbdt_100m",
        lambda: lgb.LGBMRegressor(n_estimators=35000, random_state=42),
    )


def train_lgbm_rf_10m() -> lgb.LGBMRegressor:
    return load_model(
        "lgbm_rf_10m",
        lambda: lgb.LGBMRegressor(
            boosting_type="rf",
            n_estimators=700,
            num_leaves=8000,
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

    print(f"Benchmarking naive implementation of `{name}`...")
    naive_dump_time = benchmark(base_dumps_func, model)
    naive_pickled = base_dumps_func(model)
    naive_pickled_size = len(naive_pickled)
    naive_load_time = benchmark(base_loads_func, naive_pickled)

    print(f"Benchmarking our implementation of `{name}`...")
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
    models = [
        ("sklearn rf 20M", train_sklearn_rf_20m),
        ("sklearn rf 200M", train_sklearn_rf_200m),
        ("sklearn rf 1G", train_sklearn_rf_1g),
        ("sklearn gb 2M", train_sklearn_gb_2m),
        ("lgbm gbdt 2M", train_lgbm_gbdt_2m),
        ("lgbm gbdt 5M", train_lgbm_gbdt_5m),
        ("lgbm gbdt 20M", train_lgbm_gbdt_20m),
        ("lgbm gbdt 100M", train_lgbm_gbdt_100m),
        ("lgbm rf 10M", train_lgbm_rf_10m),
    ]

    def get_dumps_args(model_name, train_func):
        if "sklearn" in model_name:
            return (model_name, train_func) + dumps_sklearn_args
        elif "lgbm" in model_name:
            return (model_name, train_func) + dumps_lgbm_args
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def get_dumps_args_lzma(model_name, train_func):
        if "sklearn" in model_name:
            return (model_name + " lzma", train_func) + dumps_sklearn_lzma_args
        elif "lgbm" in model_name:
            return (model_name + " lzma", train_func) + dumps_lgbm_lzma_args
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    models_to_benchmark = itertools.chain.from_iterable(
        [[get_dumps_args(*model), get_dumps_args_lzma(*model)] for model in models]
    )
    benchmark_results = [benchmark_model(*args) for args in models_to_benchmark]
    results_str = format_benchmarks_results_table(benchmark_results)
    with open("benchmark.md", "w") as f:
        f.write("Base results / Our results / Change\n")
        f.write(results_str)
