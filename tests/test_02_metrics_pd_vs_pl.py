import numpy as np
import pandas as pd
import polars as pl

def _align_frames_for_compare(pd_df: pd.DataFrame, pl_df: pl.DataFrame, symbol: str):
    # filter symbol and inner-join on timestamp for safe comparison
    a = pd_df[pd_df['symbol'] == symbol][['timestamp','rolling_ma','rolling_vol','rolling_mu','rolling_sharpe']].copy()
    b = pl_df.filter(pl.col('symbol') == symbol).select(['timestamp','rolling_ma','rolling_vol','rolling_mu','rolling_sharpe']).to_pandas()
    merged = a.merge(b, on='timestamp', suffixes=('_pd','_pl'))
    return merged

def test_metrics_pd_vs_pl(df_pd, df_pl):
    from metrics import compute_rolling_pandas, compute_rolling_polars
    pd_out, _, _ = compute_rolling_pandas(df_pd, window=20)
    pl_out, _, _ = compute_rolling_polars(df_pl, window=20)

    # Pick a symbol present in both
    sym = pd_out['symbol'].unique()[0]
    merged = _align_frames_for_compare(pd_out, pl_out, sym)

    # Allow small numerical tolerance
    for col in ['rolling_ma', 'rolling_vol', 'rolling_mu', 'rolling_sharpe']:
        x = merged[f'{col}_pd'].astype(float).to_numpy()
        y = merged[f'{col}_pl'].astype(float).to_numpy()
        # ignore initial NaNs where window not satisfied
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            assert np.allclose(x[mask], y[mask], rtol=1e-5, atol=1e-8), f"{col} mismatch"
