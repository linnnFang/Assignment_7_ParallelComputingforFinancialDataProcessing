# portfolio.py
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def compute_position_metrics(symbol: str, quantity: float, price: float) -> dict:
    """Compute value, volatility, and drawdown for a single position."""
    np.random.seed(abs(hash(symbol)) % (2**32))
    returns = np.random.normal(0, 0.01, 252)

    volatility = float(np.std(returns, ddof=1))
    price_path = price * np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(price_path)
    drawdown_series = (price_path - rolling_max) / rolling_max
    max_drawdown = float(drawdown_series.min())

    return {
        "symbol": symbol,
        "value": round(quantity * price, 2),
        "volatility": round(volatility, 3),
        "drawdown": round(max_drawdown, 3),
    }


def _aggregate(results_positions, results_subs):
    """Helper to compute total metrics."""
    total_value = sum(p["value"] for p in results_positions)
    total_value += sum(sp["total_value"] for sp in results_subs)

    # Weighted average volatility
    vol_weights, vol_values = [], []
    for p in results_positions:
        vol_weights.append(p["value"])
        vol_values.append(p["volatility"])
    for sp in results_subs:
        vol_weights.append(sp["total_value"])
        vol_values.append(sp["aggregate_volatility"])
    aggregate_volatility = (
        float(np.average(vol_values, weights=vol_weights))
        if sum(vol_weights) > 0
        else 0.0
    )

    # Max drawdown (worst)
    all_drawdowns = [p["drawdown"] for p in results_positions] + [
        sp["max_drawdown"] for sp in results_subs
    ]
    max_drawdown = float(min(all_drawdowns)) if all_drawdowns else 0.0

    return round(total_value, 2), round(aggregate_volatility, 3), round(max_drawdown, 3)



def aggregate_portfolio_sequential(portfolio: dict) -> dict:
    positions = []
    sub_results = []

    # Compute position metrics sequentially
    for pos in portfolio.get("positions", []):
        positions.append(
            compute_position_metrics(pos["symbol"], pos["quantity"], pos["price"])
        )

    # Recursively process sub-portfolios
    for sp in portfolio.get("sub_portfolios", []):
        sub_results.append(aggregate_portfolio_sequential(sp))

    total_value, agg_vol, max_dd = _aggregate(positions, sub_results)

    result = {
        "name": portfolio["name"],
        "total_value": total_value,
        "aggregate_volatility": agg_vol,
        "max_drawdown": max_dd,
        "positions": positions,
    }
    if sub_results:
        result["sub_portfolios"] = sub_results
    return result



def aggregate_portfolio_parallel(portfolio: dict, max_workers: int = 4) -> dict:
    positions_data = portfolio.get("positions", [])
    sub_portfolios = portfolio.get("sub_portfolios", [])
    positions = []

    # Parallel computation for positions
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                compute_position_metrics, pos["symbol"], pos["quantity"], pos["price"]
            )
            for pos in positions_data
        ]
        for f in as_completed(futures):
            positions.append(f.result())

    # Sequential recursion for sub-portfolios (parallel possible if needed)
    sub_results = [
        aggregate_portfolio_parallel(sp, max_workers=max_workers)
        for sp in sub_portfolios
    ]

    total_value, agg_vol, max_dd = _aggregate(positions, sub_results)

    result = {
        "name": portfolio["name"],
        "total_value": total_value,
        "aggregate_volatility": agg_vol,
        "max_drawdown": max_dd,
        "positions": positions,
    }
    if sub_results:
        result["sub_portfolios"] = sub_results
    return result

