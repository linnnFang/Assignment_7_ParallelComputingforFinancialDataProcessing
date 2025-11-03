# reporting.py
import time
import psutil
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import json

import os
import re


def measure_ingestion_time(file_path: str, n_runs: int = 3):
    pandas_times, polars_times = [], []

    for _ in range(n_runs):
        start = time.perf_counter()
        _ = pd.read_csv(file_path, parse_dates=['timestamp'])
        pandas_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = pl.read_csv(file_path, try_parse_dates=True)
        polars_times.append(time.perf_counter() - start)

    return np.mean(pandas_times), np.mean(polars_times)


def measure_rolling_time(data, window=20):
    if isinstance(data, pd.DataFrame):
        start = time.perf_counter()
        _ = data.groupby('symbol')['price'].rolling(window).mean().reset_index()
        return time.perf_counter() - start

    elif isinstance(data, pl.DataFrame):
        start = time.perf_counter()
        _ = data.group_by('symbol').agg(pl.col('price').rolling_mean(window))
        return time.perf_counter() - start


def measure_memory_usage(data):
    if isinstance(data, pd.DataFrame):
        return data.memory_usage(deep=True).sum() / 1e6
    elif isinstance(data, pl.DataFrame):
        return data.estimated_size() / 1e6


def measure_parallel_speed(data, func):
    if isinstance(data, pd.DataFrame):
        start = time.perf_counter()
        _ = data['price'].apply(func)
        return time.perf_counter() - start
    elif isinstance(data, pl.DataFrame):
        start = time.perf_counter()
        _ = data.select(func(pl.col('price')))
        return time.perf_counter() - start



def save_separate_comparison_charts(summary, out_dir=".", ylabel="Time / Memory"):
    """
    summary: either a pandas DataFrame with columns ['Metric','Pandas','Polars']
             or a dict convertible to such a DataFrame.
    Returns: list of saved file paths.
    """
    if isinstance(summary, dict):
        summary = pd.DataFrame(summary)

    required = {'Metric', 'Pandas', 'Polars'}
    if not required.issubset(set(summary.columns)):
        raise ValueError(f"summary must have columns {required}")

    os.makedirs(out_dir, exist_ok=True)
    saved = []

    for _, row in summary.iterrows():
        metric_name = str(row['Metric'])
        pandas_val = float(row['Pandas'])
        polars_val = float(row['Polars'])

        # One chart per metric (no subplots, as requested)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(2)
        vals = [pandas_val, polars_val]

        ax.bar(x, vals, width=0.6)  # no explicit colors/style
        ax.set_xticks(x)
        ax.set_xticklabels(['Pandas', 'Polars'])
        ax.set_ylabel(ylabel)
        ax.set_title(metric_name)

        # Optional: annotate bars
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3g}", ha='center', va='bottom')

        plt.tight_layout()

        # Safe filename from metric name
        fname = re.sub(r'[^A-Za-z0-9_-]+', '_', metric_name.strip()).lower()
        path = os.path.join(out_dir, f"{fname}.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved
def compare_performance(summary, file_path="market_data-1.csv"):
    # Ingestion
    pandas_ingest, polars_ingest = measure_ingestion_time(file_path)

    # Load once for other tests
    df_pd = pd.read_csv(file_path, parse_dates=['timestamp'])
    df_pl = pl.read_csv(file_path, try_parse_dates=True)

    # Rolling metric
    pandas_roll = measure_rolling_time(df_pd)
    polars_roll = measure_rolling_time(df_pl)

    # Memory
    pandas_mem = measure_memory_usage(df_pd)
    polars_mem = measure_memory_usage(df_pl)

    # Parallel execution (price squared)
    func = lambda x: x ** 2
    pandas_parallel = measure_parallel_speed(df_pd, func)
    polars_parallel = measure_parallel_speed(df_pl, func)

    # Summary table
    summary = pd.DataFrame({
        'Metric': ['Ingestion Time (s)', 'Rolling Mean Time (s)', 'Memory Usage (MB)', 'Parallel Speed (s)'],
        'Pandas': [pandas_ingest, pandas_roll, pandas_mem, pandas_parallel],
        'Polars': [polars_ingest, polars_roll, polars_mem, polars_parallel]
    })

    # Visualization
    '''
     fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(summary))

    ax.bar(x - bar_width/2, summary['Pandas'], bar_width, label='Pandas')
    ax.bar(x + bar_width/2, summary['Polars'], bar_width, label='Polars')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['Metric'], rotation=30, ha='right')
    ax.set_ylabel("Time / Memory")
    ax.set_title("Performance Comparison: Pandas vs Polars")
    ax.legend()
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    '''
    save_separate_comparison_charts(summary, out_dir=".", ylabel="Time / Memory")

    # Discussion
    discussion = {
        "observations": {
            "pandas": "Mature ecosystem and flexible for time-series operations.",
            "polars": "Faster ingestion and parallelized expressions, ideal for large structured data."
        },
        "tradeoffs": {
            "syntax": "Pandas feels natural for analysts; Polars requires functional-style thinking.",
            "ecosystem": "Pandas integrates with scikit-learn, matplotlib, and statsmodels more smoothly.",
            "scalability": "Polars handles multi-core and lazy evaluation better."
        }
    }

    result = {"summary_table": summary.to_dict(orient="records"), "discussion": discussion}

    
    with open("performance_summary.json", "w") as f:
        json.dump(result, f, indent=2)
