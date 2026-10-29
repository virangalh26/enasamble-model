"""Rotation Forest (Rodríguez, Kuncheva & Alonso, IEEE TPAMI 2006), written for scikit-learn.

Rotation Forest is an ensemble of decision trees in which every tree sees the data in its own rotated
coordinate system:

1. Split the features at random into disjoint subsets of `subset_size` features.
2. For each subset, draw a random sub-sample of the rows (a random non-empty subset of the classes,
   then `sample_fraction` of those rows) and fit PCA on it. Keep *all* principal components, so no
   information is thrown away; the axes are only turned.
3. Put the subset rotations on the diagonal of one block matrix R and train a decision tree on X @ R.
4. Predict by averaging the trees' class probabilities.

How it differs from the forests taught in lectures: Random Forest and Bagging make trees diverse by
re-sampling rows and features, but every tree still splits on the original axes, so a boundary like
"midday import is low *compared with* night import" needs a staircase of splits. Rotation Forest gives
every tree a different set of oblique axes (linear combinations of features), which keeps each tree
accurate while making the trees disagree in useful ways.

Before rotating, the features are put on a common scale (signed log1p, then z-score), because PCA is
sensitive to scale and several features here are ratios with long tails. Trees themselves do not care.
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


def _rotation(X, y, classes, subset_size, sample_fraction, rng):
    """One tree's block rotation matrix (p x p) built from PCA on random feature subsets."""
    n, p = X.shape
    perm = rng.permutation(p)
    R = np.zeros((p, p))
    for start in range(0, p, subset_size):
        cols = perm[start:start + subset_size]
        keep = rng.choice(classes, rng.integers(1, len(classes) + 1), replace=False)
        rows = np.flatnonzero(np.isin(y, keep))
        rows = rng.choice(rows, max(len(cols) + 1, int(len(rows) * sample_fraction)), replace=False)
        sub = X[np.ix_(rows, cols)]
        # SVD of the centred sub-sample: rows of vt are the principal axes (a full orthonormal basis).
        # (rows > len(cols) is guaranteed above, so vt is square.)
        _, _, vt = np.linalg.svd(sub - sub.mean(axis=0), full_matrices=False)
        R[np.ix_(cols, cols)] = vt.T
    return R


def _fit_one(X, y, classes, params, seed):
    rng = np.random.default_rng(seed)
    R = _rotation(X, y, classes, params["subset_size"], params["sample_fraction"], rng)
    rows = np.arange(len(X))
    if params["max_samples"] < 1.0:  # optional row sub-sampling per tree (speed); the paper uses all rows
        rows = rng.choice(rows, int(len(rows) * params["max_samples"]), replace=False)
    tree = DecisionTreeClassifier(max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
                                  max_features=params["max_features"], random_state=int(rng.integers(2**31 - 1)))
    tree.fit(X[rows] @ R, y[rows])
    return R, tree


class RotationForestClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=50, subset_size=3, sample_fraction=0.75, max_depth=None, min_samples_leaf=1,
                 max_features=None, max_samples=1.0, rotate=True, n_jobs=-1, random_state=None):
        self.n_estimators = n_estimators
        self.subset_size = subset_size
        self.sample_fraction = sample_fraction
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.rotate = rotate  # False = same trees without rotation (ablation: shows what the rotation adds)
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _scale(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.sign(X) * np.log1p(np.abs(X))
        return (Z - self.mean_) / self.std_

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        Xa = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(np.asarray(y), return_inverse=True)
        self.n_features_in_ = Xa.shape[1]
        Z = np.sign(Xa) * np.log1p(np.abs(Xa))
        self.mean_, self.std_ = Z.mean(axis=0), Z.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (Z - self.mean_) / self.std_
        params = {"subset_size": self.subset_size if self.rotate else 1, "sample_fraction": self.sample_fraction,
                  "max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf,
                  "max_features": self.max_features, "max_samples": self.max_samples}
        seeds = np.random.default_rng(self.random_state).integers(2**31 - 1, size=self.n_estimators)
        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(Xs, y_enc, np.arange(len(self.classes_)), params, s) for s in seeds)
        # With subset_size 1 each "rotation" is a sign flip of one feature: an axis-aligned tree, no rotation.
        self.rotations_ = [R for R, _ in fitted]
        self.estimators_ = [t for _, t in fitted]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        Xs = self._scale(X)
        proba = np.zeros((len(Xs), len(self.classes_)))
        for R, tree in zip(self.rotations_, self.estimators_):
            proba[:, tree.classes_] += tree.predict_proba(Xs @ R)
        return proba / len(self.estimators_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def staged_predict_proba(self, X):
        """Ensemble probabilities after 1, 2, ... trees (for the 'accuracy vs number of trees' plot)."""
        Xs = self._scale(X)
        total = np.zeros((len(Xs), len(self.classes_)))
        for k, (R, tree) in enumerate(zip(self.rotations_, self.estimators_), start=1):
            total[:, tree.classes_] += tree.predict_proba(Xs @ R)
            yield total / k
