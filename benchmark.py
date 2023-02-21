import io
import pickle
import textwrap
import time
from typing import Callable, List

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from examples.utils import load_data
from pickle_compression.lgbm_booster import dump_lgbm
from pickle_compression.sklearn_tree import dump_sklearn


def train_model_sklearn() -> RandomForestRegressor:
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(*load_data())
    return regressor


def train_model_lgbm() -> lgb.LGBMRegressor:
    regressor = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    regressor.fit(*load_data())
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
    naive_pickled = pickle.dumps(model)
    naive_pickled_size = len(naive_pickled)
    naive_load_time = benchmark(pickle.loads, naive_pickled)
    our_pickled_buf = io.BytesIO()
    dump_func(model, our_pickled_buf)
    our_pickled = our_pickled_buf.getvalue()
    our_pickled_size = len(our_pickled)
    our_load_time = benchmark(pickle.loads, our_pickled)
    return {
        "name": name,
        "baseline": {
            "size": naive_pickled_size,
            "load_time": naive_load_time,
        },
        "ours": {"size": our_pickled_size, "load_time": our_load_time},
        "change": {
            "size": naive_pickled_size / our_pickled_size,
            "load_time": our_load_time / naive_load_time,
        },
    }


def format_size(n_bytes: int) -> str:
    MiB = 1024**2
    return f"{n_bytes/MiB:.1f} MiB"


def format_time(seconds: float) -> str:
    return f"{seconds:.1f} s"


def format_change(multiple: float) -> str:
    return f"{multiple:.1f} x"


def format_benchmarks_results_table(benchmark_results: List[dict]) -> str:
    header = f"""
        | Model | Baseline Size | Our Size | Size Reduction | Baseline Loading Time | Our Loading Time | Slowdown |
        |--|--:|--:|--:|--:|--:|--:|
    """

    def format_row(results):
        column_data = [
            results["name"],
            format_size(results["baseline"]["size"]),
            format_size(results["ours"]["size"]),
            format_change(results["change"]["size"]),
            format_time(results["baseline"]["load_time"]),
            format_time(results["ours"]["load_time"]),
            format_change(results["change"]["load_time"]),
        ]
        return " | ".join(map(str, column_data))

    formatted_rows = map(format_row, benchmark_results)

    return (textwrap.dedent(header) + "\n".join(formatted_rows)).strip()


if __name__ == "__main__":
    models_to_benchmark = [
        ("sklearn `RandomForestRegressor`", train_model_sklearn, dump_sklearn),
        ("lightgbm `LGBMRegressor`", train_model_lgbm, dump_lgbm),
    ]
    benchmark_results = [benchmark_model(*args) for args in models_to_benchmark]
    print(format_benchmarks_results_table(benchmark_results))
