import pandas as pd
import polars as pl
import time
import psutil
import matplotlib.pyplot as plt
from reporting import compare_performance

def main():
    # --- Part 1: Data Ingestion ---
    start_time = time.time()
    df_pd = pd.read_csv("market_data-1.csv", parse_dates=["timestamp"])
    pandas_ingest_time = time.time() - start_time

    start_time = time.time()
    df_pl = pl.read_csv("market_data-1.csv")
    df_pl = df_pl.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )
    polars_ingest_time = time.time() - start_time

    # --- Part 2: Rolling Metrics ---
    window_size = 3

    # Pandas rolling mean
    start_time = time.time()
    df_pd["rolling_mean"] = (
        df_pd.groupby("symbol")["price"].transform(lambda x: x.rolling(window_size).mean())
    )
    pandas_roll_time = time.time() - start_time

    # Polars rolling mean
    start_time = time.time()
    roll_pl = (
        df_pl.group_by("symbol", maintain_order=True)
        .agg([
            pl.col("price").rolling_mean(window_size).alias("rolling_mean"),
            pl.col("timestamp").alias("timestamp_list"),
        ])
        .explode(["rolling_mean", "timestamp_list"])  # only explode list columns
        .rename({"timestamp_list": "timestamp"})
    )
    polars_roll_time = time.time() - start_time

    # --- Part 3: Memory Usage ---
    process = psutil.Process()
    mem_usage = process.memory_info().rss / 1024**2  # in MB

    # --- Part 4: Parallel Execution Simulation ---
    start_time = time.time()
    df_pl_par = (
        df_pl.group_by("symbol", maintain_order=True)
        .agg(pl.col("price").rolling_mean(window_size).alias("rolling_mean"))
    )
    polars_parallel_time = time.time() - start_time

    # --- Fix Nested Data Issue Before Writing ---
    for col in roll_pl.columns:
        if roll_pl[col].dtype == pl.List:
            roll_pl = roll_pl.with_columns(
                roll_pl[col].list.mean().alias(f"{col}_mean")
            ).drop(col)
        elif roll_pl[col].dtype == pl.Struct:
            for field in roll_pl[col].struct.fields:
                roll_pl = roll_pl.with_columns(
                    roll_pl[col].struct.field(field).alias(f"{col}_{field}")
                )
            roll_pl = roll_pl.drop(col)

    # Write results safely
    df_pd.to_csv("rolling_metrics_pandas.csv", index=False)
    roll_pl.write_csv("rolling_metrics_polars.csv")

    # --- Part 5: Reporting ---
    summary = {
        "pandas_ingest_time": pandas_ingest_time,
        "polars_ingest_time": polars_ingest_time,
        "pandas_roll_time": pandas_roll_time,
        "polars_roll_time": polars_roll_time,
        "polars_parallel_time": polars_parallel_time,
        "memory_usage_mb": mem_usage,
    }

    compare_performance((summary, "market_data-1.csv"))

    print("Performance summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
