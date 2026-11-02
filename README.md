# AI-Optimized Fmax Search for Test-Time Reduction

Machine-learning-assisted speed binning for semiconductor test. A gradient-boosted model predicts each
chip's maximum operating frequency (Fmax) from on-chip ring-oscillator counts, leakage, Vmin and die
position. Calibrated uncertainty turns the prediction into a per-unit search window, and a binary search
restricted to that window replaces the blind full-range search. The result is still a **measured** Fmax,
so the binning rule and guard band are unchanged.

Implements the approach from *AI-Optimized Fmax Search for Test Time Reduction* (v1.0, Verilumen Labs
hiring assessment) on the supplied synthetic dataset (6,000 units: 1,500 train / 500 calibration / 4,000 test).

## Results (held-out test split, 4,000 units, one corner)

Model: RMSE 27.7 MHz, R² 0.93, calibration σ 26.4 MHz, ±1.5σ window coverage 85%.

| Method | Mean tests | p95 | Time (ms) | Saving vs binary | Saving vs linear | Under-bin |
|---|---|---|---|---|---|---|
| Linear sweep | 40.48 | 58 | 1214 | −526% | 0% | 9.6% |
| Binary search | 6.46 | 7 | 194 | 0% | 84% | 9.0% |
| Bayesian (population prior) | 6.44 | 8 | 193 | 0.4% | 84% | 11.8% |
| ML-seeded walk | 5.32 | 10 | 160 | 18% | 87% | 9.4% |
| **ML-windowed binary** | **3.98** | 9 | 119 | **38.5%** | 90% | 9.8% |
| ML + Bayesian | 4.27 | 6 | 128 | 34% | 89% | 13.1% |
| ML confirm-bin (binning only) | 1.79 | 2 | 54 | 72% | 96% | 11.3% |

Full output: `results/benchmark_summary.csv`, `results/model_metrics.json`, `results/run_log.txt`.

**Key takeaways**
- Windowing the search with the model cuts test applications by about 38% versus binary search and 90% versus a linear sweep, with no over-bin escapes in this run.
- Using the prediction as a *starting point* (seeded walk) saves far less than using it to *bound* a binary search.
- Under-bin of about 10% for every measured method comes from the 8 MHz guard band and 10 MHz grid, not the model.
- Confirm-only binning is cheapest but does not measure Fmax, and showed 2 over-bin escapes in 4,000 units; it needs much larger samples before it could be qualified.
- Production safety needs the surrounding flow: OOD fallback (implemented via Mahalanobis distance), audit sampling, drift monitoring.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                     # unit tests for the search algorithms
python run_pipeline.py        # trains, benchmarks, writes results/
python -m streamlit run app.py
```

## Project layout

```
fmax_search/
  config.py     grid, bins, guard band, noise (Table 5)
  data.py       load + split
  model.py      GBM + calibration residuals + conformal interval + OOD check
  search.py     linear, binary, windowed (Alg. 1), Bayesian bisection (Alg. 2), confirm-only (Alg. 3)
  simulate.py   benchmark harness and metrics
  plots.py      search-trajectory figure
run_pipeline.py CLI: train -> benchmark -> results/
app.py          Streamlit dashboard
tests/          pytest tests
```

## Method notes and limitations
- Test noise is drawn fresh on every pass/fail application (σ_test = 4 MHz); time per step is 30 ms (25 apply + 5 switch).
- Reference bin = bin of the true Fmax with no guard band (this matches the dataset's `true_bin_mhz`); flow bins apply the 8 MHz guard band.
- Data is synthetic. Numbers are illustrative and must be re-established on real product data.
- Single voltage/temperature corner only; cross-corner reuse (Alg. 4) is not implemented.
- The dataset's precomputed columns (`fmax_pred_mhz`, `tests_windowed`, ...) are not used for training; the pipeline recomputes everything from features.
- My ML-seeded walk starts at the lower window edge and differs in detail from the paper's walk, so its numbers (5.3 tests) differ from the paper's (6.2).
