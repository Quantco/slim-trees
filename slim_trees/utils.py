import io
import warnings

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from slim_trees import __version__ as slim_trees_version


def check_version(version: str):
    if version != slim_trees_version:
        warnings.warn(
            f"Version mismatch: slim_trees version {slim_trees_version} "
            f"does not match version {version} of the model."
        )


def pyarrow_table_to_bytes(table: pa.Table) -> bytes:
    """
    Given a pyarrow Table, return a .parquet file as bytes.
    """
    stream = pa.BufferOutputStream()
    pq.write_table(
        table, stream, compression="lz4"
    )  # TODO: investigate different effects of compression
    return stream.getvalue().to_pybytes()


def pq_bytes_to_df(bytes_: bytes) -> pd.DataFrame:
    """
    Given a .parquet file as bytes, return a pandas DataFrame.
    """
    return pd.read_parquet(io.BytesIO(bytes_))
