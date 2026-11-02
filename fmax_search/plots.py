import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from . import search as S


def trajectories(true_f, lo, hi, seed=1):
    """Figure-3 style trajectories for one unit: linear, binary, ML-windowed."""
    runs = []
    for title, fn in [
        ("Linear sweep", lambda t: S.linear_sweep(t)),
        ("Binary search", lambda t: S.binary_search(t)),
        ("ML-windowed binary", lambda t: S.windowed_binary(t, lo, hi)),
    ]:
        t = S.NoisyTester(true_f, np.random.default_rng(seed))
        res = fn(t)
        runs.append((title, t.log, res))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (title, log, res) in zip(axes, runs):
        ax.axhline(true_f, ls="--", c="k", lw=.8)
        if title.startswith("ML"):
            ax.axhspan(lo, hi, color="green", alpha=.12)
        xs = np.arange(1, len(log) + 1)
        fs = [f for f, _ in log]
        ax.plot(xs, fs, c="grey", lw=.8)
        ax.scatter(xs, fs, c=["#4c9a4c" if ok else "#d9534f" for _, ok in log], zorder=3, s=25)
        ax.set(title=f"{title} ({len(log)} tests)", xlabel="Test #")
    axes[0].set_ylabel("Test frequency (MHz)")
    fig.tight_layout()
    return fig, runs
