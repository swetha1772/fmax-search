"""Streamlit dashboard: python -m streamlit run app.py"""
import json
import pandas as pd
import streamlit as st
from fmax_search.config import FEATURES, TARGET
from fmax_search.data import load_dataset, split
from fmax_search.model import FmaxModel
from fmax_search.plots import trajectories
from fmax_search.simulate import METHODS, run_all, summarize

st.set_page_config(page_title="AI-Optimized Fmax Search", page_icon="", layout="wide")


@st.cache_resource
def load():
    df = load_dataset()
    train, calib, test = split(df)
    return df, test, FmaxModel().fit(train, calib)


@st.cache_data(show_spinner="Running search simulation...")
def benchmark(n_units, k, seed):
    _, test, model = load()
    return summarize(run_all(test.head(n_units), model, k=k, seed=seed))


df, test, model = load()

st.title("⚡ AI-Optimized Fmax Search for Test-Time Reduction")
st.caption("Predict each chip's Fmax from on-chip monitors, then search only inside the predicted window.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Single-unit demo", "Benchmark", "Dataset"])

with tab1:
    y = model.predict(test)
    err = test[TARGET].values - y
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test units", len(test))
    c2.metric("RMSE (MHz)", f"{(err ** 2).mean() ** .5:.1f}")
    c3.metric("R²", f"{1 - (err ** 2).sum() / ((test[TARGET] - test[TARGET].mean()) ** 2).sum():.3f}")
    c4.metric("Calibration σ (MHz)", f"{model.sigma:.1f}")
    st.scatter_chart(pd.DataFrame({"Measured Fmax (MHz)": test[TARGET].values,
                                   "Predicted Fmax (MHz)": y}),
                     x="Measured Fmax (MHz)", y="Predicted Fmax (MHz)", height=380)
    st.markdown(
        "**How it works:** a gradient-boosted model maps ring-oscillator counts, leakage, Vmin and die "
        "position to Fmax. Calibrated residuals give a per-unit window `ŷ ± 1.5σ`. A binary search "
        "inside that window replaces the blind full-range search, so the result is still a *measured* "
        "Fmax and the binning rule is unchanged. Units that look out-of-distribution fall back to the full search.")

with tab2:
    ids = test.unit_id.tolist()
    uid = st.selectbox("Unit", ids, index=0)
    row = test[test.unit_id == uid]
    k = st.slider("Window width k (± k·σ)", 0.5, 3.0, 1.5, 0.25)
    seed = st.number_input("Noise seed", 0, 9999, 1)
    yh, lo, hi = model.predict_interval(row, k=k)
    st.write(f"True Fmax **{row[TARGET].iloc[0]:.1f} MHz** · predicted **{yh[0]:.1f} MHz** · "
             f"window **[{lo[0]:.0f}, {hi[0]:.0f}] MHz** · OOD: **{bool(model.is_ood(row)[0])}**")
    fig, runs = trajectories(row[TARGET].iloc[0], lo[0], hi[0], seed=int(seed))
    st.pyplot(fig)
    st.table(pd.DataFrame([{"Method": t, "Tests": len(log), "Measured Fmax (MHz)": res} for t, log, res in runs]))

with tab3:
    n = st.slider("Units to simulate", 200, len(test), 1000, 200)
    kb = st.slider("Window width k", 0.5, 3.0, 1.5, 0.25, key="kb")
    summary = benchmark(n, kb, 0)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index("Method")["Mean tests"])
    st.caption("Under-bin ≈ 10% for all measured methods comes from the 8 MHz guard band and 10 MHz grid, not the model.")

with tab4:
    st.write(f"{len(df):,} units · split: {df.split.value_counts().to_dict()}")
    st.dataframe(df.head(200), use_container_width=True)
    st.write(df[FEATURES + [TARGET]].describe().T)
