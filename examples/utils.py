import os
import tempfile
import time
from itertools import product
from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from pickle_compression.pickling import dump_compressed, load_compressed


def generate_dataset(
    n_samples: int = 50000, n_features: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dataset with 50000 samples and 100 features.

    Returns:
        X_train (np.array): (0.8 * n_samples) x n_features
        X_test (np.array): (0.2 * n_samples) x n_features
        y_train (np.array): 0.8 * n_samples
        y_test (np.array): 0.2 * n_samples
    """
    X, y = make_regression(  # noqa: N806
        n_samples=n_samples,
        n_features=n_features,
        n_informative=50,
        n_targets=1,
        shuffle=True,
        random_state=42,
    )

    # make some columns categorical
    for i in range(0, 100, 10):
        X[:, i] = X[:, i].astype("int")

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def evaluate_compression_performance(
    model: Any, dump: Callable, print_performance: bool = True
):
    compressions = ["no", "lzma", "bz2", "gzip"]
    performance = []
    for compression, dump_function in product(compressions, [None, dump]):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/model"
            start = time.time()
            dump_compressed(model, path, compression, dump_function)
            dump_time = time.time() - start
            start = time.time()
            load_compressed(path, compression)
            load_time = time.time() - start
            size = os.path.getsize(path)
        performance += [
            {
                "compression": compression,
                "dump_function": dump_function.__name__ if dump_function else None,
                "size": f"{size / 2 ** 20:.2f} MB",
                "dump_time": f"{dump_time:.3f} s",
                "load_time": f"{load_time:.3f} s",
            }
        ]
    df = pd.DataFrame(performance)
    if print_performance:
        print(df.to_string(index=False))
    return df


def evaluate_prediction_difference(
    model: Any, model_compressed: Any, x: np.ndarray, print_performance: bool = True
) -> np.ndarray:
    y_pred = model.predict(x)
    y_pred_new = model_compressed.predict(x)
    diff = np.abs(y_pred - y_pred_new)
    if print_performance:
        print("Prediction difference:")
        print(f"Max: {diff.max()}")
        print(f"Avg: {diff.mean()}")
        print(f"Std: {diff.std()}")
    return diff
