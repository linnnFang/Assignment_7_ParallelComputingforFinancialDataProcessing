import pandas as pd
import numpy as np
import polars as pl
import time
import psutil

def compute_rolling_pandas(df: pd.DataFrame, window: int = 20):
    """
    Compute rolling MA (on price), rolling std & mean (on returns), 
    and rolling Sharpe (rf=0, not annualized).
    Returns: (DataFrame, elapsed_time, delta_mem_mb)
    """

    # Ensure sorted by symbol & time
    if "timestamp" in df.columns:
        df = df.sort_values(["symbol", "timestamp"]).copy()

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2
    t0 = time.perf_counter()

    # Group by symbol
    grouped = df.groupby("symbol", group_keys=False)

    # Step 1: compute returns per symbol
    df["ret"] = grouped["price"].pct_change()

    # Step 2: rolling metrics
    df["rolling_ma"] = grouped["price"].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    df["rolling_vol"] = grouped["ret"].transform(
        lambda x: x.rolling(window, min_periods=window).std(ddof=1)
    )
    df["rolling_mu"] = grouped["ret"].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    df["rolling_sharpe"] = (df["rolling_mu"] / df["rolling_vol"]).replace([np.inf, -np.inf], np.nan)

    # Step 3: clean up
    elapsed = time.perf_counter() - t0
    mem_after = process.memory_info().rss / 1024**2
    delta_mem = mem_after - mem_before

    print(f"[Pandas] Time: {elapsed:.3f}s | ΔMemory: {delta_mem:.2f} MB | window={window}")
    return df, elapsed, delta_mem

def compute_rolling_polars(df: pl.DataFrame, window: int = 20):
    """
    Compute rolling MA (on price), rolling mean/std (on returns), and rolling Sharpe (rf=0).
    Equivalent to compute_rolling_pandas().
    Returns: (pl.DataFrame, elapsed_time, delta_mem_mb)
    """

    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2
    t0 = time.perf_counter()

    # Ensure proper column names
    assert set(["timestamp", "symbol", "price"]).issubset(set(df.columns)), "missing required columns"

    # Step 1. Compute returns per symbol
    df = (
        df.sort(["symbol", "timestamp"])
          .with_columns([
              (pl.col("price") / pl.col("price").shift(1) - 1)
              .over("symbol")
              .alias("ret")
          ])
    )

    # Step 2. Rolling metrics
    df = df.with_columns([
        # Rolling mean of price
        pl.col("price")
          .rolling_mean(window_size=window, min_periods=window)
          .over("symbol")
          .alias("rolling_ma"),

        # Rolling mean and std of returns
        pl.col("ret")
          .rolling_mean(window_size=window, min_periods=window)
          .over("symbol")
          .alias("rolling_mu"),

        pl.col("ret")
          .rolling_std(window_size=window, min_periods=window)
          .over("symbol")
          .alias("rolling_vol"),
    ])

    # Step 3. Rolling Sharpe
    df = df.with_columns(
        (pl.col("rolling_mu") / pl.col("rolling_vol"))
        .alias("rolling_sharpe")
    )

    # Step 4. Clean up + timing
    elapsed = time.perf_counter() - t0
    mem_after = process.memory_info().rss / 1024**2
    delta_mem = mem_after - mem_before

    print(f"[Polars] Time: {elapsed:.3f}s | ΔMemory: {delta_mem:.2f} MB | window={window}")
    return df, elapsed, delta_mem
