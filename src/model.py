"""Poverty targeting model: NGBoost predicts a distribution for log consumption per person.

NGBoost (Duan et al., 2020) fits the parameters (mu, log sigma) of a Normal distribution with gradient
boosting on the *natural* gradient of the log-likelihood, so every household gets its own spread.
That is the standard proxy-means-test set-up (predict log consumption from observable assets),
with an uncertainty attached:

    P(poor) = P(log c < log z) = Phi((log z - mu) / sigma),   z = official poverty line

A household is flagged for support when P(poor) is at least `threshold` (chosen on training data only).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.features import to_matrix


@dataclass
class PovertyTargeter:
    model: object                 # fitted ngboost.NGBRegressor with Dist=Normal
    columns: list                 # model matrix columns after one-hot encoding
    poverty_line: float
    threshold: float = 0.5
    info: dict = field(default_factory=dict)

    def matrix(self, households: pd.DataFrame) -> pd.DataFrame:
        return to_matrix(households, self.columns)

    def distribution(self, X: pd.DataFrame):
        d = self.model.pred_dist(X)
        return np.asarray(d.params["loc"]), np.asarray(d.params["scale"])

    def p_poor_from_matrix(self, X: pd.DataFrame) -> np.ndarray:
        mu, sigma = self.distribution(X)
        return norm.cdf((np.log(self.poverty_line) - mu) / sigma)

    def predict(self, households: pd.DataFrame, interval: float = 0.8) -> pd.DataFrame:
        X = self.matrix(households)
        mu, sigma = self.distribution(X)
        zq = norm.ppf(0.5 + interval / 2)
        p = norm.cdf((np.log(self.poverty_line) - mu) / sigma)
        return pd.DataFrame({
            "p_poor": p,
            "pred_pc_consumption": np.exp(mu),
            "pc_low": np.exp(mu - zq * sigma),
            "pc_high": np.exp(mu + zq * sigma),
            "sigma": sigma,
            "flag_for_support": (p >= self.threshold).astype(int),
        }, index=households.index)
