import pathlib
import tempfile

from sklearn.ensemble import RandomForestRegressor
from utils import (
    evaluate_compression_performance,
    evaluate_prediction_difference,
    generate_dataset,
)

from slim_trees import dump_sklearn_compressed
from slim_trees.pickling import load_compressed
from slim_trees.sklearn_tree import dump


def train_model() -> RandomForestRegressor:
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    X, y = generate_dataset(n_samples=10000)
    regressor.fit(X, y)
    return regressor


if __name__ == "__main__":
    model = train_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "model.pkl"
        dump_sklearn_compressed(model, path, "no")
        model_compressed = load_compressed(path, "no")

    evaluate_prediction_difference(
        model, model_compressed, generate_dataset(n_samples=10000)[0]
    )
    evaluate_compression_performance(model, dump)
