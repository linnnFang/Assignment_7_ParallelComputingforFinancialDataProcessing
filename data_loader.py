# data_loader.py
import time
import psutil
import pandas as pd
import polars as pl


def _mem_mb():
    """Return current process memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024**2


def load_pandas(csv_path: str, use_pyarrow: bool = True, dtypes: dict | None = None):
    """
    Load the market data using pandas.
    Parses the timestamp column, sorts by symbol and timestamp,
    and sets timestamp as the index.
    Also profiles ingestion time and memory usage.
    """
    t0, m0 = time.perf_counter(), _mem_mb()

    read_kw = dict(parse_dates=["timestamp"])
    if use_pyarrow:
        read_kw.update(engine="pyarrow")  # faster parser if available
    if dtypes:
        read_kw.update(dtype=dtypes)

    df = pd.read_csv(csv_path, **read_kw)

    required_cols = ["timestamp", "symbol", "price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert symbol to category to reduce memory and improve groupby speed
    '''
    df["symbol"] = df["symbol"].astype("category")
    '''
    # Sort by symbol (primary key) and timestamp (secondary key)
    # Then set timestamp as index for time-series operations
    df = df.sort_values(["symbol", "timestamp"]).set_index("timestamp")

    prof = {
        "impl": "pandas",
        "rows": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "ingest_time_s": time.perf_counter() - t0,
        "delta_mem_mb": _mem_mb() - m0,
    }
    return df, prof


def load_polars(csv_path: str, dtypes: dict | None = None):
    """
    Load the same CSV using polars.
    Ensures equivalent parsing logic (timestamp parsing, sorting, schema check).
    """
    t0, m0 = time.perf_counter(), _mem_mb()

    df = pl.read_csv(csv_path, try_parse_dates=True, dtypes=dtypes)

    # Ensure timestamp column is actually Datetime
    ts_dtype = df.schema.get("timestamp")
    if ts_dtype == pl.Utf8:
        df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, strict=False))
    elif ts_dtype == pl.Date:
        df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

    required = {"timestamp", "symbol", "price"}
    if not required.issubset(set(df.columns)):
        missing = list(required - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by symbol and timestamp (same logic as pandas)
    df = df.sort(["symbol", "timestamp"])

    prof = {
        "impl": "polars",
        "rows": int(df.height),
        "n_symbols": int(df.get_column("symbol").n_unique()),
        "ingest_time_s": time.perf_counter() - t0,
        "delta_mem_mb": _mem_mb() - m0,
    }
    return df, prof


def sanity_check_equivalence(df_pd: pd.DataFrame, df_pl: pl.DataFrame, sample: int = 5) -> bool:
    """
    Compare pandas and polars outputs to confirm equivalent parsing.
    Checks row count, symbol count, and a few head/tail samples.
    """
    if len(df_pd) != df_pl.height:
        return False

    n_sym_pd = df_pd["symbol"].nunique()
    n_sym_pl = int(df_pl.get_column("symbol").n_unique())
    if n_sym_pd != n_sym_pl:
        return False

    syms = list(df_pd["symbol"].cat.categories)[:sample] if hasattr(df_pd["symbol"], "cat") else sorted(df_pd["symbol"].unique())[:sample]
    for s in syms:
        pd_sub = df_pd[df_pd["symbol"] == s]
        pl_sub = df_pl.filter(pl.col("symbol") == s)
        if pd_sub.empty or pl_sub.is_empty():
            continue
        pd_head = pd_sub.head(3).reset_index()[["timestamp", "price"]]
        pl_head = pl_sub.head(3).select(["timestamp", "price"]).to_pandas()
        if not _rough_equal(pd_head, pl_head):
            return False
        pd_tail = pd_sub.tail(3).reset_index()[["timestamp", "price"]]
        pl_tail = pl_sub.tail(3).select(["timestamp", "price"]).to_pandas()
        if not _rough_equal(pd_tail, pl_tail):
            return False
    return True


def _rough_equal(df1: pd.DataFrame, df2: pd.DataFrame, tol: float = 1e-9) -> bool:
    """Check equality between two small DataFrames (allow tiny float differences)."""
    import numpy as np
    if len(df1) != len(df2):
        return False
    if not (df1["timestamp"].values == df2["timestamp"].values).all():
        return False
    return np.allclose(df1["price"].values, df2["price"].values, rtol=0, atol=tol)

