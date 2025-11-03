import os
import sys
import json
import pandas as pd
import polars as pl
import pytest

# Ensure imports resolve to this repo
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CSV_PATH = os.path.join(REPO_ROOT, "market_data-1.csv")
PORTFOLIO_JSON = os.path.join(REPO_ROOT, "portfolio_structure-1.json")

@pytest.fixture(scope="session")
def csv_path():
    assert os.path.exists(CSV_PATH), f"Missing CSV at {CSV_PATH}"
    return CSV_PATH

@pytest.fixture(scope="session")
def portfolio_path():
    assert os.path.exists(PORTFOLIO_JSON), f"Missing portfolio JSON at {PORTFOLIO_JSON}"
    return PORTFOLIO_JSON

@pytest.fixture(scope="session")
def df_pd(csv_path):
    from data_loader import load_pandas
    df, _ = load_pandas(csv_path)
    # keep it small for faster tests (subset a handful of symbols)
    small_syms = df['symbol'].unique()[:3]
    return df[df['symbol'].isin(small_syms)].copy()

@pytest.fixture(scope="session")
def df_pl(csv_path):
    from data_loader import load_polars
    df, _ = load_polars(csv_path)
    # keep it small for faster tests (subset a handful of symbols)
    small_syms = df.get_column('symbol').unique()[:3]
    return df.filter(pl.col('symbol').is_in(small_syms)).clone()
