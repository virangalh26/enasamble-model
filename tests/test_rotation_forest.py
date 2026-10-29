"""Unit tests for the Rotation Forest implementation (no trained model or data needed)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.rotation_forest import RotationForestClassifier, _rotation


def oblique_data(n=2000, seed=0):
    """Two classes split by the diagonal x0 + x1 = 0: easy for rotated trees, a staircase for axis-aligned ones."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    return X, (X.f0 + X.f1 > 0).astype(int).to_numpy()


def test_rotation_is_orthonormal_block_matrix():
    X, y = oblique_data()
    R = _rotation(X.to_numpy(), y, np.array([0, 1]), subset_size=3, sample_fraction=0.75,
                  rng=np.random.default_rng(1))
    assert R.shape == (6, 6)
    assert np.allclose(R.T @ R, np.eye(6), atol=1e-8)  # a pure rotation: no information lost
    assert (np.count_nonzero(R, axis=0) <= 3).all()     # each new axis mixes only one subset of features


def test_probabilities_are_valid_and_reproducible():
    X, y = oblique_data()
    a = RotationForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(X, y).predict_proba(X)
    b = RotationForestClassifier(n_estimators=10, random_state=0, n_jobs=1).fit(X, y).predict_proba(X)
    assert a.shape == (len(X), 2) and np.allclose(a.sum(1), 1) and np.array_equal(a, b)


def test_rotation_beats_the_same_trees_without_rotation_on_an_oblique_boundary():
    X, y = oblique_data()
    Xt, yt = oblique_data(seed=1)
    kw = dict(n_estimators=20, max_depth=4, random_state=0, n_jobs=1)
    rotated = (RotationForestClassifier(**kw).fit(X, y).predict(Xt) == yt).mean()
    plain = (RotationForestClassifier(rotate=False, **kw).fit(X, y).predict(Xt) == yt).mean()
    assert rotated > plain + 0.02


def test_staged_probabilities_end_at_the_full_ensemble():
    X, y = oblique_data(500)
    m = RotationForestClassifier(n_estimators=5, random_state=0, n_jobs=1).fit(X, y)
    stages = list(m.staged_predict_proba(X))
    assert len(stages) == 5 and np.allclose(stages[-1], m.predict_proba(X))


def test_is_a_scikit_learn_classifier():
    m = RotationForestClassifier(n_estimators=3, subset_size=2)
    assert clone(m).get_params()["subset_size"] == 2
    with pytest.raises(Exception):
        m.predict_proba(np.zeros((1, 6)))  # not fitted yet
