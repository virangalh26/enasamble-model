"""Experiment tracking and model registry on Azure Machine Learning (MLflow).

Logs one training as an MLflow run (with a nested child run for every model compared), saves the
explanations and the packaged model, and registers it as a new version of `poverty-targeter`.
It reads the files written by `src.train` / `src.explain`, so it never retrains anything, and it is
idempotent: a model whose SHA-256 is already registered is only re-tagged.

Usage (MLFLOW_TRACKING_URI must point at the Azure ML workspace; auth comes from `az login`/OIDC):
    python -m src.track --stage candidate                 # after retraining
    python -m src.track --stage production --image-tag X  # after deploying
"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

MODEL_NAME = "poverty-targeter"
EXPERIMENT = "poverty-targeter"
MODEL_FILE = Path("models/poverty_targeter.joblib")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def data_version() -> str:
    """md5 of the processed dataset recorded by DVC (the data version this model was trained on)."""
    lock = Path("dvc.lock")
    if not lock.exists():
        return "unknown"
    import yaml
    for out in yaml.safe_load(lock.read_text())["stages"].get("build", {}).get("outs", []):
        if out["path"].endswith("households.parquet"):
            return out.get("md5", "unknown")
    return "unknown"


def find_version(client: MlflowClient, model_hash: str):
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:  # registry empty / model not created yet
        return None
    for v in versions:
        if (v.tags or {}).get("model_sha256") == model_hash:
            return v
    return None


def log_training(model_hash: str, git_sha: str) -> str:
    meta = json.load(open("models/metadata.json"))
    comp = pd.read_csv("reports/model_comparison.csv")
    best = json.load(open("reports/best_params.json"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"ngboost-{model_hash[:8]}") as run:
        mlflow.set_tags({"model_sha256": model_hash, "git_sha": git_sha, "dataset": "HIES 2019 public 25% sample (DCS)",
                         "data_md5": data_version(), "kind": "final-model"})
        mlflow.log_params({"threshold": meta["threshold"], "n_features": len(meta["features"]),
                           "algorithm": "ngboost_normal_log_consumption", "poverty_line": meta["poverty_line"],
                           **{f"ngb_{k}": v for k, v in best["NGBoost"].items()}, "evaluation_unit": "household"})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in meta["test"].items()})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in meta["test_regression"].items()})

        for row in comp.to_dict(orient="records"):
            with mlflow.start_run(run_name=row["model"], nested=True):
                mlflow.set_tags({"family": row["family"], "model_sha256": model_hash})
                if row["model"].startswith("NGBoost (Normal"):
                    mlflow.log_params(best["NGBoost"])
                mlflow.log_metrics({k: float(v) for k, v in row.items() if k.startswith(("cv_", "test_"))})
                mlflow.log_metric("train_seconds", float(row["seconds"]))

        for path in ["models/metadata.json", "reports/metrics.json", "reports/model_comparison.csv",
                     "reports/best_params.json", "reports/fairness.csv", "models/reference_stats.json", "params.yaml"]:
            if Path(path).exists():
                mlflow.log_artifact(path, "reports")
        if Path("reports/explain").exists():
            mlflow.log_artifacts("reports/explain", "explain")

        # Package the targeter as an MLflow pyfunc model (saved locally, then uploaded as run artifacts).
        from src.mlflow_model import PovertyTargeterModel
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.pyfunc.save_model(
                path=f"{tmp}/model", python_model=PovertyTargeterModel(),
                artifacts={"targeter": str(MODEL_FILE), "metadata": "models/metadata.json"},
                code_paths=["src"], pip_requirements="requirements-app.txt")
            mlflow.log_artifacts(f"{tmp}/model", "model")
        return run.info.run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["candidate", "production"], default="candidate")
    ap.add_argument("--git-sha", default=os.environ.get("GITHUB_SHA", "local"))
    ap.add_argument("--image-tag", default="")
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    a = ap.parse_args()
    if not a.tracking_uri:
        raise SystemExit("Set MLFLOW_TRACKING_URI (see infra/azure.env).")
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_registry_uri(a.tracking_uri)
    client = MlflowClient()

    model_hash = sha256(MODEL_FILE)
    version = find_version(client, model_hash)
    if version is None:
        run_id = log_training(model_hash, a.git_sha)
        version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME,
                                        tags={"model_sha256": model_hash, "git_sha": a.git_sha})
        print(f"Logged run {run_id} and registered {MODEL_NAME} v{version.version}")
    else:
        print(f"Model {model_hash[:12]} already registered as {MODEL_NAME} v{version.version}")

    client.set_model_version_tag(MODEL_NAME, version.version, "stage", a.stage)
    if a.image_tag:
        client.set_model_version_tag(MODEL_NAME, version.version, "image_tag", a.image_tag)
    if a.stage == "production":  # only one version is tagged production at a time
        for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if v.version != version.version and (v.tags or {}).get("stage") == "production":
                client.set_model_version_tag(MODEL_NAME, v.version, "stage", "archived")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"### Model registry\n\n`{MODEL_NAME}` **v{version.version}** tagged `{a.stage}`"
                     f"{f' (image `{a.image_tag}`)' if a.image_tag else ''} — sha256 `{model_hash[:12]}`\n\n")


if __name__ == "__main__":
    main()
