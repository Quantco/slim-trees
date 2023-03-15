import pathlib
import tempfile

from sklearn.ensemble import RandomForestRegressor

from benchmark import train_model_sklearn
from utils import (
    evaluate_compression_performance,
    evaluate_prediction_difference,
    generate_dataset,
)

from slim_trees import dump_sklearn_compressed
from slim_trees.pickling import load_compressed, get_pickled_size
from slim_trees.sklearn_tree import dump_sklearn


def train_model() -> RandomForestRegressor:
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    X, y = generate_dataset(n_samples=10000)
    regressor.fit(X, y)
    return regressor


if __name__ == "__main__":
    model = train_model_sklearn()
    # model = train_model()

    print(f"No compress: {get_pickled_size(model, compression='no') / (2 ** 20)} MB")
    # print(f"No compress lzma: {get_pickled_size(model, compression='lzma') / (2 ** 20)} MB")
    print(f"compress old: {get_pickled_size(model, compression='no', dump_function=dump_sklearn) / (2 ** 20)} MB")
    print(f"compress old lzma: {get_pickled_size(model, compression='lzma', dump_function=dump_sklearn) / (2 ** 20)} MB")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "model.pkl"
        dump_sklearn_compressed(model, path, "no")
        model_compressed = load_compressed(path, "no")

    evaluate_prediction_difference(
        model, model_compressed, generate_dataset(n_samples=10000)[0]
    )
    evaluate_compression_performance(model, dump_sklearn)
