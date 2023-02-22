import pathlib
import tempfile

from sklearn.ensemble import RandomForestRegressor
from sklearn_tree import dump_sklearn

from examples.utils import (
    evaluate_compression_performance,
    evaluate_prediction_difference,
    generate_dataset,
)
from pickle_compression import dump_sklearn_compressed
from pickle_compression.pickling import load_compressed


def train_model() -> RandomForestRegressor:
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train, _, y_train, _ = generate_dataset(n_samples=10000)  # noqa: N806
    regressor.fit(X_train, y_train)
    return regressor


if __name__ == "__main__":
    model = train_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "model.pkl"
        dump_sklearn_compressed(model, path, "no")
        model_compressed = load_compressed(path, "no")

    evaluate_prediction_difference(
        model, model_compressed, generate_dataset(n_samples=10000)[1]
    )
    evaluate_compression_performance(model, dump_sklearn)
