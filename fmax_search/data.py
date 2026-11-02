import pandas as pd
from .config import FEATURES, TARGET


def load_dataset(path="data/synthetic_fmax_dataset.csv"):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET, "split"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def split(df):
    return (df[df.split == "train"], df[df.split == "calibration"], df[df.split == "test"])
