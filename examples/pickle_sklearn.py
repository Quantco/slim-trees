from sklearn.ensemble import RandomForestRegressor
from sklearn_tree import dump_sklearn

from examples.utils import load_data, print_model_size


def train_model():
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(*load_data())
    return regressor


model = train_model()

print_model_size(model, dump_sklearn)
