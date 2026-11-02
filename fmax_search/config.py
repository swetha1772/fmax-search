"""Constants matching Table 5 of the technical document."""
import numpy as np

GRID_MIN, GRID_MAX, GRID_STEP = 1600, 2400, 10      # MHz
GRID = np.arange(GRID_MIN, GRID_MAX + GRID_STEP, GRID_STEP, dtype=float)  # 81 points
SIGMA_TEST = 4.0            # MHz, test repeatability
T_APPLY_MS, T_SWITCH_MS = 25.0, 5.0
BINS = np.array([1800, 1900, 2000, 2100, 2200])
GUARD_BAND = 8.0            # MHz measurement guard band
WINDOW_K = 1.5              # window = y_hat +/- K * sigma
CONFIRM_K = 1.5             # confirm-only starting offset

RO_COLS = [f"ro_{i:02d}" for i in range(1, 11)]
FEATURES = RO_COLS + ["iddq_norm", "vmin_norm", "die_x", "die_y", "radius_sq"]
TARGET = "fmax_true_mhz"
