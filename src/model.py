"""Household-level rooftop solar detector built on a day-level classifier.

Each household-day is scored by the day model; a household's score is the mean of its day scores
(soft voting over days). The household is flagged as having solar when that mean exceeds a threshold
chosen on training data only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features import HOURS, make_features


@dataclass
class SolarDetector:
    day_model: object
    feature_names: list
    threshold: float = 0.5
    info: dict = field(default_factory=dict)

    def features(self, days: pd.DataFrame) -> pd.DataFrame:
        return make_features(days[HOURS])[self.feature_names]

    def predict_day_proba(self, days: pd.DataFrame) -> np.ndarray:
        return self.day_model.predict_proba(self.features(days))[:, 1]

    def predict_household(self, days: pd.DataFrame, hh_col: str = "hh") -> pd.DataFrame:
        p = self.predict_day_proba(days)
        out = (pd.DataFrame({hh_col: days[hh_col].values, "p_day": p})
               .groupby(hh_col)["p_day"].agg(p_solar="mean", n_days="size"))
        out["has_solar_pred"] = (out["p_solar"] >= self.threshold).astype(int)
        return out
