import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
