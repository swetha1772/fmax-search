import numpy as np
from fmax_search import search as S
from fmax_search.config import GRID


def exact(true_f):
    """Noise-free tester so results are deterministic."""
    return S.NoisyTester(true_f, np.random.default_rng(0), sigma_test=0.0)


def expected(true_f):
    ok = GRID[GRID <= true_f]
    return None if len(ok) == 0 else float(ok.max())


def test_linear_and_binary_find_exact_fmax():
    for f in [1605, 1800, 1999.9, 2063, 2399, 2400]:
        assert S.linear_sweep(exact(f)) == expected(f)
        assert S.binary_search(exact(f)) == expected(f)


def test_windowed_correct_even_when_window_is_wrong():
    f = 2063
    for lo, hi in [(2040, 2080), (1900, 1950), (2200, 2300), (1600, 2400)]:
        assert S.windowed_binary(exact(f), lo, hi) == expected(f)


def test_windowed_uses_fewer_tests_with_good_window():
    t_w, t_b = exact(2063), exact(2063)
    S.windowed_binary(t_w, 2030, 2090)
    S.binary_search(t_b)
    assert t_w.n < t_b.n


def test_assign_bin_guard_band():
    assert S.assign_bin(2008.0) == 2000
    assert S.assign_bin(2007.0) == 1900   # below 2000 + 8 MHz guard
    assert S.assign_bin(1700.0) == 0
