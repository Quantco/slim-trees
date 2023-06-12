import warnings

from slim_trees import __version__ as slim_trees_version


def check_version(version: str):
    if version != slim_trees_version:
        warnings.warn(
            f"Version mismatch: slim_trees version {slim_trees_version} "
            f"does not match version {version} of the model.",
            stacklevel=2,
        )
