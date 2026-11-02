"""Benchmark every search strategy on held-out units (Table 6 / Figure 4)."""
import numpy as np
import pandas as pd
from . import search as S
from .config import CONFIRM_K, TARGET, WINDOW_K


def reference_bin(fmax_true):
    """Dataset convention: bin of the true Fmax with no guard band."""
    return S.assign_bin(fmax_true, guard=0.0)


def run_method(name, df, model, k=WINDOW_K, seed=0):
    y, lo, hi = model.predict_interval(df, k=k)
    ood = model.is_ood(df)
    sig = model.sigma
    rng = np.random.default_rng(seed)
    rows = []
    for i, (true_f, yh, l, h, o) in enumerate(zip(df[TARGET].values, y, lo, hi, ood)):
        t = S.NoisyTester(true_f, rng)
        fmax, bin_ = None, None
        if name == "Linear sweep":
            fmax = S.linear_sweep(t)
        elif name == "Binary search":
            fmax = S.binary_search(t)
        elif name == "Bayesian (population prior)":
            fmax = S.bayesian_bisection(t, 2000.0, 110.0)
        elif name == "ML-seeded walk":
            fmax = S.seeded_walk(t, l) if not o else S.binary_search(t)
        elif name == "ML-windowed binary":
            fmax = S.windowed_binary(t, l, h) if not o else S.binary_search(t)
        elif name == "ML + Bayesian":
            fmax = S.bayesian_bisection(t, yh, sig) if not o else S.binary_search(t)
        elif name == "ML confirm-bin (binning only)":
            bin_ = S.confirm_only(t, yh, sig, k=CONFIRM_K) if not o else S.assign_bin(S.binary_search(t))
        else:
            raise ValueError(name)
        if bin_ is None:
            bin_ = S.assign_bin(fmax)
        rows.append(dict(method=name, tests=t.n, time_ms=t.time_ms,
                         fmax=fmax, bin=bin_, ref_bin=reference_bin(true_f),
                         true_fmax=true_f, fallback=bool(o) and name.startswith('ML')))
    return pd.DataFrame(rows)


METHODS = ["Linear sweep", "Binary search", "Bayesian (population prior)",
           "ML-seeded walk", "ML-windowed binary", "ML + Bayesian",
           "ML confirm-bin (binning only)"]


def summarize(results):
    binary_mean = results[results.method == "Binary search"].tests.mean()
    linear_mean = results[results.method == "Linear sweep"].tests.mean()
    out = []
    for m, g in results.groupby("method", sort=False):
        err = (g.fmax - g.true_fmax).abs() if g.fmax.notna().any() else np.nan
        out.append({
            "Method": m,
            "Mean tests": round(g.tests.mean(), 2),
            "p95 tests": int(np.percentile(g.tests, 95)),
            "Time (ms)": round(g.time_ms.mean(), 1),
            "Saving vs binary (%)": round(100 * (1 - g.tests.mean() / binary_mean), 1),
            "Saving vs linear (%)": round(100 * (1 - g.tests.mean() / linear_mean), 1),
            "|error| (MHz)": None if g.method.iloc[0].endswith("(binning only)") else round(float(err.mean()), 1),
            "Under-bin (%)": round(100 * (g.bin < g.ref_bin).mean(), 1),
            "Over-bin escapes": int((g.bin > g.ref_bin).sum()),
            "Fallback (%)": round(100 * g.fallback.mean(), 2),
        })
    return pd.DataFrame(out)


def run_all(df, model, k=WINDOW_K, seed=0, methods=METHODS):
    return pd.concat([run_method(m, df, model, k=k, seed=seed) for m in methods],
                     ignore_index=True)
