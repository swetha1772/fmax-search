"""Fmax regression model with calibrated uncertainty and an OOD check."""
import numpy as np
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import FEATURES, RO_COLS, TARGET


class FmaxModel:
    def __init__(self, random_state=0):
        self.gbm = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=4,
            l2_regularization=1.0, random_state=random_state)
        self.sigma = None          # std of calibration residuals
        self.q90 = None            # split-conformal 90% half-width
        self._mu = self._prec = self._ood_thr = None

    def fit(self, train_df, calib_df):
        self.gbm.fit(train_df[FEATURES], train_df[TARGET])
        resid = calib_df[TARGET].values - self.gbm.predict(calib_df[FEATURES])
        self.sigma = float(resid.std(ddof=1))
        n = len(resid)
        level = min(1.0, np.ceil((n + 1) * 0.90) / n)
        self.q90 = float(np.quantile(np.abs(resid), level))
        # Mahalanobis OOD check on the ring-oscillator vector (Section 8.2)
        X = train_df[RO_COLS].values
        self._mu = X.mean(0)
        self._prec = np.linalg.inv(np.cov(X, rowvar=False))
        self._ood_thr = chi2.ppf(0.999, df=X.shape[1])
        return self

    def predict(self, df):
        return self.gbm.predict(df[FEATURES])

    def predict_interval(self, df, k=1.5):
        y = self.predict(df)
        return y, y - k * self.sigma, y + k * self.sigma

    def is_ood(self, df):
        d = df[RO_COLS].values - self._mu
        m2 = np.einsum("ij,jk,ik->i", d, self._prec, d)
        return m2 > self._ood_thr
