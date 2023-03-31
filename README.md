# Slim Trees

[![CI](https://github.com/pavelzw/slim-trees/actions/workflows/ci.yml/badge.svg)](https://github.com/pavelzw/slim-trees/actions/workflows/ci.yml)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/slim-trees?logoColor=white&logo=conda-forge)](https://anaconda.org/conda-forge/slim-trees)
[![pypi-version](https://img.shields.io/pypi/v/slim-trees.svg?logo=pypi&logoColor=white)](https://pypi.org/project/slim-trees)
[![python-version](https://img.shields.io/pypi/pyversions/slim-trees?logoColor=white&logo=python)](https://pypi.org/project/slim-trees)

A python package for efficient pickling.

## Installation

```bash
pip install slim-trees
# or
mamba install slim-trees -c conda-forge
```

## Usage

Using `slim-trees` does not affect your training pipeline.
Simply replace calls to `pickle.dumps` with `slim_trees.dump_sklearn_compressed`:

```python
# example, you can also use other Tree-based models
from sklearn.tree import DecisionTreeClassifier
from slim_trees import dump_sklearn_compressed

# load training data
X, y = ...
model = DecisionTreeClassifier()
model.fit(X, y)

dump_sklearn_compressed(model, "model.pkl")
```

Later, you can load the model using `pickle.load` as usual.

```python
import pickle

model = pickle.load("model.pkl")
```

## Development Installation

You can install the package in development mode using:

```bash
git clone git@github.com:pavelzw/slim-trees.git
cd slim-trees

# create and activate a fresh environment named slim_trees
# see environment.yml for details
mamba env create
conda activate slim_trees

pre-commit install
pip install --no-build-isolation -e .
```
