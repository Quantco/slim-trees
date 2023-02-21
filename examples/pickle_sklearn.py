from sklearn.ensemble import RandomForestRegressor
from sklearn_tree import dump_sklearn

from examples.utils import evaluate_compression_performance, load_data


def train_model() -> RandomForestRegressor:
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(*load_data())
    return regressor


model = train_model()

evaluate_compression_performance(model, dump_sklearn)
