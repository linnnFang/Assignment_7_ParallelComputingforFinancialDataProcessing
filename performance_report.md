# Performance Report — Assignment 7

This report summarizes **ingestion**, **rolling analytics**, and **parallel execution** benchmarks,
comparing **pandas** and **polars**, and **threading** vs **multiprocessing**.

> Hardware/OS: _Fill in your machine details here (CPU model, cores, RAM, OS)_.  
> Dataset: `market_data-1.csv` (provided).  
> Window: 20. Risk‑free rate assumed 0.  

---

## 1) Ingestion

| Library | Mean Wall Time (s) | Notes |
| --- | ---: | --- |
| pandas | _0.0806_ | `read_csv` (pyarrow if available), parse dates |
| polars | _0.0036_ | `scan_csv`/`read_csv` with eager mode |

**Observation.** Polars typically ingests faster due to multithreaded parsing and efficient memory layout.

---

## 2) Rolling Analytics (per symbol)

Rolling metrics:
- **MA(20)** on price
- **Vol(20)** and **Mu(20)** on returns
- **Sharpe = Mu / Vol** (rf=0)

| Library | Time (s) | Comment |
| --- | ---: | --- |
| pandas | _0.0314_ | GroupBy + rolling windows |
| polars | _0.0039_ | Expression API uses native parallelism |

**Takeaway.** Polars wins for wide, columnar workloads; pandas remains idiomatic and flexible.

---

## 3) Threading vs Multiprocessing (per‑symbol rolling)

| Mode | Workers | Total Time (s) | CPU Utilization (qualitative) | Memory Δ (MB) |
| --- | ---: | ---: | --- | ---: |
| ThreadPool | 4 | _20.600132_ | Limited by GIL for CPU‑bound work | ~0 |
| ProcessPool | 4 | _5.9_ | True parallel CPU usage | +N (per‑proc overhead) |

**When to prefer processes:** CPU‑bound NumPy/pandas work.  
**When threads are fine:** I/O bound steps (ingestion, network, disk) or when using GIL‑free libraries.

---

## 4) Portfolio Aggregation

- Parallel map of **position metrics** (value/vol/drawdown)
- Recursive reduce for **sub‑portfolio totals**, **weighted vol**, and **max drawdown**

| Implementation | Total Value | Aggregate Vol | Max Drawdown | Time (s) |
| --- | ---: | ---: | ---: | ---: |
| Sequential | _match parallel_ | _match parallel_ | _match parallel_ | _0.02999_ |
| Parallel (ProcessPool) | _same_ | _same_ | _same_ | _0.00614_ |

**Deterministic RNG** seeded by symbol ensures consistent results across runs.

---

## 5) Visuals

- `performance_comparison.png` — bar chart of ingestion + rolling times.  
- Rolling traces (AAPL) are plotted by `reporting.py` to validate shape and correctness.

---

## 6) Discussion — Trade‑offs

- **Syntax & Ergonomics.** Pandas is familiar; Polars favors an expression‑style, great for pipelines.
- **Ecosystem.** Pandas integrates tightly with sklearn/statsmodels/matplotlib; Polars is catching up quickly.
- **Performance.** Polars generally faster for ingestion and columnar transforms; pandas can be competitive with numba/c‑extensions.
- **Parallelism.** For CPU‑bound tasks, prefer **multiprocessing** with pandas. Polars can leverage internal parallelism already.
- **Memory.** Processes add overhead from data duplication. Use shared formats (Arrow, memory maps) or chunked workflows to mitigate.
- **Scalability.** Consider **lazy Polars** and **out‑of‑core** techniques for very large datasets.

---

## 7) Reproduction

Run:

```bash
python -c "import reporting; reporting.compare_performance('market_data-1.csv', symbol_for_plot='AAPL')"
```

This produces the JSON summary and chart. Insert your actual measured numbers above.
