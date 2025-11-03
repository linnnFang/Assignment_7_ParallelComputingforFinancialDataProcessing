# parallel.py
import time
import psutil
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def rolling_one_symbol(df, symbol, window=20):
    """Compute rolling mean/std/sharpe (rf=0) for one symbol.
    Works whether timestamp is a column named 'timestamp' or the index.
    Returns columns: ['timestamp','symbol','price','ma','vol','sharpe']
    """
    import numpy as np
    import pandas as pd

    # slice one symbol and sort by time
    data = df[df['symbol'] == symbol].copy()

    # unify timestamp: prefer column 'timestamp'; else use index
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp')
        ts = data['timestamp'].reset_index(drop=True)
    else:
        # assume datetime index
        data = data.sort_index()
        ts = pd.Series(data.index, name='timestamp').reset_index(drop=True)

    price = data['price'].reset_index(drop=True)

    # returns on this symbol only
    ret = price.pct_change()

    # rolling stats (full windows only)
    ma  = price.rolling(window, min_periods=window).mean()
    vol = ret.rolling(window, min_periods=window).std(ddof=1)
    mu  = ret.rolling(window, min_periods=window).mean()
    sharpe = (mu / vol).replace([np.inf, -np.inf], np.nan)

    # build output
    out = pd.DataFrame({
        'timestamp': ts,
        'symbol': symbol,
        'price': price,
        'ma': ma,
        'vol': vol,
        'sharpe': sharpe
    })

    return out


def run_parallel(df, mode="thread", window=20, max_workers=None):
    """
    Run rolling computation across symbols using threading or multiprocessing.
    Returns: (df_out, elapsed_time, delta_mem_mb)
    """
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024**2
    t0 = time.perf_counter()

    symbols = df["symbol"].unique().tolist()
    if mode == "thread":
        Executor = ThreadPoolExecutor
    elif mode == "process":
        Executor = ProcessPoolExecutor
    else:
        raise ValueError("mode must be 'thread' or 'process'")

    results = []
    with Executor(max_workers=max_workers) as ex:
        futures = [ex.submit(rolling_one_symbol, df, s, window) for s in symbols]
        for f in as_completed(futures):  # avoids blocking on first completed
            results.append(f.result())

    df_out = pd.concat(results, ignore_index=True)
    elapsed = time.perf_counter() - t0
    mem_after = process.memory_info().rss / 1024**2
    delta_mem = mem_after - mem_before

    print(f"[{mode.capitalize()}] Time: {elapsed:.3f}s | ΔMemory: {delta_mem:.2f} MB")
    return df_out, elapsed, delta_mem