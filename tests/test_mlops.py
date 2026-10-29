import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.drift import psi, psi_report, reference_stats
from src.features import HOURS, make_features


def days(n, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return make_features(pd.DataFrame(rng.gamma(2.0, 0.1 * scale, (n, 24)), columns=HOURS))


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == pytest.approx(0.0)


def test_drift_detected_only_when_distribution_shifts():
    ref = reference_stats(days(5000))
    same = psi_report(days(5000, seed=1), ref)
    shifted = psi_report(days(5000, scale=3.0, seed=2), ref)
    assert (same.psi < 0.1).all()
    assert shifted.set_index("feature").loc["total_kwh", "status"] == "drift"


def _write(tmp_path, name, acc, auc=0.99, f1=0.97):
    p = tmp_path / name
    p.write_text(json.dumps({"test": {"hh_accuracy": acc, "hh_auc": auc, "hh_f1": f1}}))
    return str(p)


@pytest.mark.parametrize("new_acc, ok", [(0.992, True), (0.990, True), (0.980, False)])
def test_quality_gate(tmp_path, monkeypatch, new_acc, ok):
    from src import quality_gate
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"quality_gate": {"min_hh_accuracy": 0.985, "max_drop": 0.003}}))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr("sys.argv", ["qg", "--new", _write(tmp_path, "new.json", new_acc),
                                     "--baseline", _write(tmp_path, "base.json", 0.9918), "--params", str(params)])
    with pytest.raises(SystemExit) as e:
        quality_gate.main()
    assert (e.value.code == 0) == ok
