"""Train, calibrate, benchmark, and write results/ (run: python run_pipeline.py)."""
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fmax_search.config import TARGET
from fmax_search.data import load_dataset, split
from fmax_search.model import FmaxModel
from fmax_search.simulate import run_all, summarize


def main(csv="data/synthetic_fmax_dataset.csv", out="results"):
    df = load_dataset(csv)
    train, calib, test = split(df)
    model = FmaxModel().fit(train, calib)

    pred = model.predict(test)
    err = test[TARGET].values - pred
    _, lo, hi = model.predict_interval(test, k=1.5)
    metrics = {
        "n_train": len(train), "n_calibration": len(calib), "n_test": len(test),
        "rmse_mhz": float(np.sqrt((err ** 2).mean())),
        "nrmse_pct_of_range": float(100 * np.sqrt((err ** 2).mean()) / (df[TARGET].max() - df[TARGET].min())),
        "mae_mhz": float(np.abs(err).mean()),
        "r2": float(1 - (err ** 2).sum() / ((test[TARGET] - test[TARGET].mean()) ** 2).sum()),
        "sigma_calibration_mhz": model.sigma,
        "window_coverage_k1.5": float(((test[TARGET] >= lo) & (test[TARGET] <= hi)).mean()),
        "conformal_90_halfwidth_mhz": model.q90,
        "conformal_90_coverage": float((np.abs(err) <= model.q90).mean()),
        "ood_fallback_rate": float(model.is_ood(test).mean()),
    }
    print(json.dumps(metrics, indent=2))

    results = run_all(test, model)
    summary = summarize(results)
    print(summary.to_string(index=False))

    summary.to_csv(f"{out}/benchmark_summary.csv", index=False)
    with open(f"{out}/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(model, f"{out}/model.joblib")

    # Figures
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(test[TARGET], pred, s=4, alpha=.4)
    lim = [test[TARGET].min(), test[TARGET].max()]
    ax.plot(lim, lim, "k-", lw=.8)
    ax.set(xlabel="Measured Fmax (MHz)", ylabel="Predicted Fmax (MHz)",
           title=f"GBM, RMSE = {metrics['rmse_mhz']:.1f} MHz")
    fig.tight_layout(); fig.savefig(f"{out}/pred_vs_measured.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(summary["Method"], summary["Mean tests"], color="#2f5d8a")
    ax.invert_yaxis(); ax.set_xlabel("Mean test applications per part")
    fig.tight_layout(); fig.savefig(f"{out}/tests_per_method.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
