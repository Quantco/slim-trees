# Slim Trees

[![CI](https://github.com/pavelzw/pickle-compression/actions/workflows/ci.yml/badge.svg)](https://github.com/pavelzw/pickle-compression/actions/workflows/ci.yml)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/pickle-compression?logoColor=white&logo=conda-forge)](https://anaconda.org/conda-forge/pickle-compression)
[![pypi-version](https://img.shields.io/pypi/v/pickle-compression.svg?logo=pypi&logoColor=white)](https://pypi.org/project/pickle-compression)
[![python-version](https://img.shields.io/pypi/pyversions/pickle-compression?logoColor=white&logo=python)](https://pypi.org/project/pickle-compression)

A python package for efficient pickling.

## Installation

You can install the package in development mode using:

```bash
git clone git@github.com:pavelzw/pickle-compression.git
cd pickle-compression

# create and activate a fresh environment named slim_trees
# see environment.yml for details
mamba env create
conda activate slim_trees

pre-commit install
pip install --no-build-isolation -e .
```
