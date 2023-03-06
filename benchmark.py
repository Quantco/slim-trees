import io
import pickle
import textwrap
import time
from pathlib import Path
from typing import Callable, List

import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from examples.utils import generate_dataset
from slim_trees.lgbm_booster import dump_lgbm


def train_gb_sklearn() -> GradientBoostingRegressor:
    regressor = GradientBoostingRegressor(n_estimators=2000, random_state=42)
    regressor.fit(*generate_dataset(n_samples=10000))
    return regressor


def train_model_sklearn() -> RandomForestRegressor:
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(*generate_dataset(n_samples=10000))
    return regressor


def train_gbdt_lgbm() -> lgb.LGBMRegressor:
    regressor = lgb.LGBMRegressor(n_estimators=2000, random_state=42)
    regressor.fit(*generate_dataset(n_samples=10000))
    return regressor


def train_gbdt_large_lgbm() -> lgb.LGBMRegressor:
    regressor = lgb.LGBMRegressor(n_estimators=20000, random_state=42)
    regressor.fit(*generate_dataset(n_samples=10000))
    return regressor


def load_rf_lgbm() -> lgb.LGBMRegressor:
    if Path("examples/benchmark_models/rf_lgbm.model").exists():
        return lgb.LGBMRegressor(model_file="examples/rf_lgbm.model")

    else:
        regressor = lgb.LGBMRegressor(
            boosting_type="rf",
            n_estimators=100,
            num_leaves=1000,
            random_state=42,
            bagging_freq=5,
            bagging_fraction=0.5,
        )
        regressor.fit(*generate_dataset(n_samples=10000))
        regressor.booster_.save_model("examples/benchmark_models/rf_lgbm.model")
        return regressor


def benchmark(func: Callable, *args, **kwargs) -> float:
    times = []
    for _ in range(10):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return min(times)


def benchmark_model(name, train_func, dump_func) -> dict:
    model = train_func()

    naive_dump_time = benchmark(pickle.dumps, model)
    naive_pickled = pickle.dumps(model)
    naive_pickled_size = len(naive_pickled)
    naive_load_time = benchmark(pickle.loads, naive_pickled)

    our_dump_time = benchmark(dump_func, model, io.BytesIO())
    our_pickled_buf = io.BytesIO()
    dump_func(model, our_pickled_buf)
    our_pickled = our_pickled_buf.getvalue()
    our_pickled_size = len(our_pickled)
    our_load_time = benchmark(pickle.loads, our_pickled)
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


if __name__ == "__main__":
    models_to_benchmark = [
        # ("`RandomForestRegressor`", train_model_sklearn, dump_sklearn),
        # ("`GradientBoostingRegressor`", train_gb_sklearn, dump_sklearn),
        # ("`LGBMRegressor gbdt`", train_gbdt_lgbm, dump_lgbm),
        # ("`LGBMRegressor gbdt large`", train_gbdt_large_lgbm, dump_lgbm),
        ("`LGBMRegressor rf`", load_rf_lgbm, dump_lgbm),
    ]
    benchmark_results = [benchmark_model(*args) for args in models_to_benchmark]
    print("Base results / Our results / Change")
    print(format_benchmarks_results_table(benchmark_results))
