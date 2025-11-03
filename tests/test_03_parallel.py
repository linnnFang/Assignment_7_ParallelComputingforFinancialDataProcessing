import numpy as np
import pandas as pd

def test_rolling_one_symbol_matches_pandas(df_pd):
    from parallel import rolling_one_symbol
    from metrics import compute_rolling_pandas
    # pick one symbol
    sym = df_pd['symbol'].unique()[0]
    pd_roll, _, _ = compute_rolling_pandas(df_pd[df_pd['symbol']==sym].copy(), window=20)
    # compute via helper
    out = rolling_one_symbol(df_pd, sym, window=20)

    # map column names for comparison
    merged = pd.DataFrame({
        'timestamp': pd_roll['timestamp'].to_numpy(),
        'ma_pd': pd_roll['rolling_ma'].to_numpy(),
        'vol_pd': pd_roll['rolling_vol'].to_numpy(),
        'sharpe_pd': (pd_roll['rolling_mu'] / pd_roll['rolling_vol']).replace([np.inf,-np.inf], np.nan).to_numpy(),
    }).merge(out[['timestamp','ma','vol','sharpe']], on='timestamp', how='inner')

    mask = np.isfinite(merged['ma_pd']) & np.isfinite(merged['ma'])
    if mask.any():
        assert np.allclose(merged.loc[mask,'ma_pd'], merged.loc[mask,'ma'], rtol=1e-5, atol=1e-8)

    mask = np.isfinite(merged['vol_pd']) & np.isfinite(merged['vol'])
    if mask.any():
        assert np.allclose(merged.loc[mask,'vol_pd'], merged.loc[mask,'vol'], rtol=1e-5, atol=1e-8)

    mask = np.isfinite(merged['sharpe_pd']) & np.isfinite(merged['sharpe'])
    if mask.any():
        assert np.allclose(merged.loc[mask,'sharpe_pd'], merged.loc[mask,'sharpe'], rtol=1e-5, atol=1e-8)

def test_thread_vs_process_consistency(df_pd):
    from parallel import run_parallel
    # Run a small subset for speed
    small = df_pd[df_pd['symbol'].isin(df_pd['symbol'].unique()[:3])].copy()
    thread_out, _, _ = run_parallel(small, mode='thread', window=20, max_workers=2)
    proc_out, _, _ = run_parallel(small, mode='process', window=20, max_workers=2)

    # Groupby to mean on timestamp+symbol to guard against ordering
    key = ['symbol','timestamp']
    tm = thread_out.groupby(key)[['ma','vol','sharpe']].mean().reset_index()
    pm = proc_out.groupby(key)[['ma','vol','sharpe']].mean().reset_index()
    merged = tm.merge(pm, on=key, suffixes=('_t','_p'))

    for col in ['ma','vol','sharpe']:
        t = merged[f'{col}_t'].to_numpy()
        p = merged[f'{col}_p'].to_numpy()
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.any():
            assert np.allclose(t[mask], p[mask], rtol=1e-5, atol=1e-8), f"{col} differs between thread/process"
