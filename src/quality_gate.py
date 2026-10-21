"""Quality gate for retraining: block a new model that is worse than the current one.

Compares reports/metrics.json (new) with the metrics of the model currently on main.
Exit code 1 fails the GitHub Actions job, so a worse model never reaches a pull request.

Usage:
    python -m src.quality_gate --new reports/metrics.json --baseline /tmp/baseline_metrics.json
"""
import argparse
import json
import os
import sys

import yaml

CHECKS = ["hh_accuracy", "hh_auc", "hh_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="reports/metrics.json")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--params", default="params.yaml")
    a = ap.parse_args()
    gate = yaml.safe_load(open(a.params))["quality_gate"]
    new = json.load(open(a.new))["test"]
    base = json.load(open(a.baseline))["test"]

    rows, ok = [], True
    for k in CHECKS:
        passed = new[k] >= base[k] - gate["max_drop"]
        ok &= passed
        rows.append(f"| {k} | {base[k]:.4f} | {new[k]:.4f} | {new[k] - base[k]:+.4f} | {'✅' if passed else '❌'} |")
    floor = new["hh_accuracy"] >= gate["min_hh_accuracy"]
    ok &= floor
    rows.append(f"| hh_accuracy ≥ {gate['min_hh_accuracy']} | — | {new['hh_accuracy']:.4f} | — | {'✅' if floor else '❌'} |")

    report = ("### Quality gate: " + ("PASSED ✅" if ok else "FAILED ❌") + "\n\n"
              "| metric | current model | new model | change | ok |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        open(summary, "a").write(report + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
