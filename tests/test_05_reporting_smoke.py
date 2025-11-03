import os
import json

def test_reporting_smoke(csv_path, tmp_path):
    from reporting import measure_ingestion_time, measure_rolling_time
    # light run for smoke testing
    pandas_t, polars_t = measure_ingestion_time(csv_path, n_runs=1)
    assert pandas_t and polars_t
    aapl = measure_rolling_time(csv_path, 'AAPL', window=20, n_runs=1)
    assert aapl and isinstance(aapl, dict) and 'pandas' in aapl and 'polars' in aapl
