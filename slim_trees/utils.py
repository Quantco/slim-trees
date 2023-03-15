import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# table_enc = pa.Table.from_arrays([*table], schema=target_schema)
# pq.write_table(table_enc, "private_/lgb1.parquet.lz4", compression="lz4")
# table_bytes = pyarrow_table_to_bytes(table_enc)
# pickle.dump(table_bytes, open("private_/byte_dumps/lgb1.parquet.pkl", "wb"))

def pyarrow_table_to_bytes(table: pa.Table) -> bytes:
    """
    Given a pyarrow Table, return a .parquet file as bytes.
    """
    stream = pa.BufferOutputStream()
    pq.write_table(
        table, stream, compression="ZSTD"
    )  # TODO: investigate different effects of compression


    return stream.getvalue().to_pybytes()


def pq_bytes_to_df(bytes_: bytes) -> pd.DataFrame:
    """
    Given a .parquet file as bytes, return a pandas DataFrame.
    """
    return pd.read_parquet(io.BytesIO(bytes_))
