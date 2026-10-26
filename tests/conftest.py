"""Shared fixture: run the whole pipeline once on synthetic HIES-shaped files (tests never see real data)."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.hies import build_households, load_raw
from tests.fake_hies import write_fake_hies

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    raw = write_fake_hies(tmp / "raw", n_households=900)
    params = yaml.safe_load(open(ROOT / "params.yaml"))
    hh = build_households(load_raw(raw), params["data"]["poverty_line"], 1.0, 1.0)
    data = tmp / "households.parquet"
    hh.to_parquet(data, index=False)
    models, reports = tmp / "models", tmp / "reports"
    subprocess.run([sys.executable, "-m", "src.train", "--data", str(data), "--models-dir", str(models),
                    "--reports-dir", str(reports), "--quick"], cwd=ROOT, check=True, capture_output=True)
    return {"raw": raw, "households": hh, "data": data, "models": models, "reports": reports}
