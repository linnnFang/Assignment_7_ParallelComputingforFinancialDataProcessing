# Assignment 7 — Parallel Time‑Series Analytics (Finance)

Design and implement a Python module that processes large‑scale financial time‑series data using **parallel computing** techniques.
You will explore the tradeoffs between **threading** and **multiprocessing**, benchmark **pandas vs polars** for performance,
and apply concurrency to accelerate analytics such as rolling metrics, signal generation, and portfolio aggregation.

## Repo Layout

```
data_loader.py          # Pandas/Polars ingestion + equivalence checks
metrics.py              # Rolling analytics (MA, volatility, Sharpe) in pandas & polars
parallel.py             # ThreadPool vs ProcessPool (per‑symbol rolling)
portfolio.py            # Parallel recursive aggregation for portfolio trees
reporting.py            # Profiling helpers + summary table + saved artifacts
main.py                 # Orchestration entry point (optional)
market_data-1.csv       # Provided sample dataset
portfolio_structure-1.json
tests/                  # Pytest suite (correctness + parity + smoke tests)
performance_report.md   # Benchmarks, charts, and analysis (this repo’s write‑up)
```

## Quickstart

### 1) Environment

- Python 3.10+
- Recommended: create a fresh virtualenv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt  # if provided
# or install minimal deps:
pip install pandas polars psutil memory_profiler matplotlib pytest
```

### 2) Run the suite

```bash
pytest -q
```

The tests validate:

- Pandas/Polars **ingestion equivalence** (`data_loader.py`).
- **Rolling metrics** parity between pandas & polars (`metrics.py`).
- **Threading vs multiprocessing** parity for per‑symbol rolling (`parallel.py`).
- **Portfolio aggregation** (sequential vs parallel) consistency (`portfolio.py`).
- A light **reporting smoke test** (`reporting.py`).

### 3) Reproduce performance figures

The `reporting.py` helpers save a `performance_summary.json` and charts.

```bash
python -c "import reporting; reporting.compare_performance('market_data-1.csv', symbol_for_plot='AAPL')"
```

Artifacts created:

- `performance_summary.json` — summary table + narrative discussion.
- `performance_comparison.png` — bar chart of key timings.
- `rolling_metrics_pandas.csv` / `rolling_metrics_polars.csv` — cached rolling outputs.

## What to Read

- `metrics.py`: clean reference implementation for rolling MA/vol/Sharpe in both libraries.
- `parallel.py`: per‑symbol function + ThreadPool/ProcessPool orchestration (with timing/memory).
- `portfolio.py`: deterministic, parallel aggregation with recursive combination rules.
- `reporting.py`: minimal but reproducible micro‑benchmarks for ingestion/rolling/parallel speed.

## Notes on Design Decisions

- **Windowed analytics:** compute returns per‑symbol, then rolling stats with identical definitions across libs.
- **Parallel split:** per‑symbol sharding makes workloads embarrassingly parallel; it also sidesteps pandas’ GIL limits by using processes for CPU‑bound work.
- **Determinism:** portfolio metrics use symbol‑seeded RNG so unit tests are stable.
- **Tolerance:** tests allow small floating‑point differences (rtol=1e‑5) due to implementation differences.

## Troubleshooting

- If `polars` is missing: `pip install polars`
- If on Apple Silicon and see BLAS/Accelerate warnings: safe to ignore for this assignment.
- If multiprocessing hangs on Windows in interactive sessions, run with `python -m pytest -q` from a terminal.
