import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.drift import DRIFT_FEATURES, psi, psi_report, reference_stats


def households(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f: rng.gamma(3.0, 1.0 + shift, n) for f in DRIFT_FEATURES})


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == pytest.approx(0.0)


def test_drift_detected_only_when_distribution_shifts():
    ref = reference_stats(households(5000))
    same = psi_report(households(5000, seed=1), ref)
    shifted = psi_report(households(5000, shift=1.5, seed=2), ref)
    assert (same.psi < 0.1).all()
    assert (shifted.status == "drift").all()


def _write(tmp_path, name, pr, auc=0.85, f1=0.55):
    p = tmp_path / name
    p.write_text(json.dumps({"test": {"pr_auc": pr, "roc_auc": auc, "f1": f1}}))
    return str(p)


@pytest.mark.parametrize("new_pr, ok", [(0.62, True), (0.59, True), (0.55, False), (0.30, False)])
def test_quality_gate(tmp_path, monkeypatch, new_pr, ok):
    from src import quality_gate
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"quality_gate": {"min_test_pr_auc": 0.4, "max_drop": 0.02}}))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr("sys.argv", ["qg", "--new", _write(tmp_path, "new.json", new_pr),
                                     "--baseline", _write(tmp_path, "base.json", 0.60), "--params", str(params)])
    with pytest.raises(SystemExit) as e:
        quality_gate.main()
    assert (e.value.code == 0) == ok
