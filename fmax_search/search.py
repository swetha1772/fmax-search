"""Fmax search algorithms (Sections 2, 3 and 6 of the technical document).

Every search runs against a NoisyTester: pass(f) is True when
true_fmax + N(0, sigma_test) >= f, i.e. noise is drawn on every application.
Each function returns (fmax_mhz, log) where log is a list of (freq, passed).
fmax_mhz is None when even the lowest grid point fails.
"""
import numpy as np
from scipy.stats import norm
from .config import (BINS, CONFIRM_K, GRID, GUARD_BAND, SIGMA_TEST, T_APPLY_MS,
                     T_SWITCH_MS)


class NoisyTester:
    def __init__(self, true_fmax, rng, sigma_test=SIGMA_TEST):
        self.f, self.rng, self.s = true_fmax, rng, sigma_test
        self.log = []

    def __call__(self, freq):
        ok = bool(self.f + self.rng.normal(0, self.s) >= freq)
        self.log.append((float(freq), ok))
        return ok

    @property
    def n(self):
        return len(self.log)

    @property
    def time_ms(self):
        return self.n * (T_APPLY_MS + T_SWITCH_MS)


def _bsearch(test, lo, hi):
    """Highest passing index in [lo, hi] (monotone assumption); lo-1 if none."""
    best = lo - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if test(GRID[mid]):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def _result(idx):
    return None if idx < 0 else float(GRID[idx])


def linear_sweep(test):
    idx = -1
    for i, f in enumerate(GRID):
        if test(f):
            idx = i
        else:
            break
    return _result(idx)


def binary_search(test):
    return _result(_bsearch(test, 0, len(GRID) - 1))


def windowed_binary(test, lo_mhz, hi_mhz):
    """Algorithm 1: binary search restricted to the model window, with
    upward extension (edge hit) and downward fallback (all fail)."""
    n = len(GRID)
    lo = int(np.clip(np.searchsorted(GRID, lo_mhz, "left"), 0, n - 1))
    hi = int(np.clip(np.searchsorted(GRID, hi_mhz, "right") - 1, lo, n - 1))
    best = _bsearch(test, lo, hi)
    if best == hi and hi < n - 1:                     # hit upper edge -> extend up
        stride, base = 1, hi
        while base + stride <= n - 1 and test(GRID[base + stride]):
            base += stride
            stride *= 2
        top = min(base + stride - 1, n - 1)
        best = _bsearch(test, base + 1, top) if base + 1 <= top else base
        best = max(best, base)
    elif best == lo - 1:                              # everything failed -> go below
        best = _bsearch(test, 0, lo - 1) if lo > 0 else -1
    return _result(best)


def seeded_walk(test, start_mhz):
    """Naive ML-seeded linear walk (the baseline the paper argues against)."""
    i = int(np.clip(np.searchsorted(GRID, start_mhz, "left"), 0, len(GRID) - 1))
    if test(GRID[i]):
        while i + 1 < len(GRID) and test(GRID[i + 1]):
            i += 1
        return float(GRID[i])
    while i > 0:
        i -= 1
        if test(GRID[i]):
            return float(GRID[i])
    return None


def bayesian_bisection(test, mu, sigma, sigma_test=SIGMA_TEST,
                       target_width=15.0, n_max=12):
    """Algorithm 2: probabilistic bisection; returns the 5th percentile."""
    F = np.arange(GRID[0] - 100, GRID[-1] + 100, 1.0)
    p = norm.pdf(F, mu, sigma)
    p /= p.sum()
    for _ in range(n_max):
        cdf = np.cumsum(p)
        if F[np.searchsorted(cdf, 0.95)] - F[np.searchsorted(cdf, 0.05)] <= target_width:
            break
        f = GRID[np.abs(GRID - F[np.searchsorted(cdf, 0.5)]).argmin()]
        like = norm.cdf((F - f) / sigma_test)
        p = p * (like if test(f) else 1 - like)
        p /= p.sum()
    return float(F[np.searchsorted(np.cumsum(p), 0.05)])


def assign_bin(fmax, guard=GUARD_BAND):
    """Highest bin with Fmax >= bin + guard band, else 'reject' (0)."""
    if fmax is None:
        return 0
    ok = BINS[BINS + guard <= fmax]
    return int(ok.max()) if len(ok) else 0


def confirm_only(test, y_hat, sigma, k=CONFIRM_K, guard=GUARD_BAND):
    """Algorithm 3: confirm the predicted bin at its lower edge (bin only)."""
    cand = np.where(BINS <= y_hat - k * sigma)[0]
    b = int(cand.max()) if len(cand) else -1
    if b >= 0 and test(BINS[b] + guard):
        while b + 1 < len(BINS) and y_hat + 2 * sigma > BINS[b + 1] and test(BINS[b + 1] + guard):
            b += 1
        return int(BINS[b])
    b = max(b, 0)
    while b > 0:
        b -= 1
        if test(BINS[b] + guard):
            return int(BINS[b])
    return 0
