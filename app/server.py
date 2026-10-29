"""Rooftop Solar Detector: a small web app made of this Python server and one HTML page.

This file loads the trained Rotation Forest and answers JSON requests. index.html, next to it, is
the whole screen: it calls the /api/... URLs below with fetch() and draws the results.
It uses only the Python standard library for the web part (no Streamlit or Flask).

Run on this computer (opens the browser):
    python app/server.py
Run in the container on Azure:
    python app/server.py --host 0.0.0.0 --port 8501 --no-browser
"""
import argparse
import io
import json
import math
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.data import readings_to_days  # noqa: E402
from src.features import HOURS, make_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "solar_detector.joblib"
META_PATH = ROOT / "models" / "metadata.json"
PAGE_FILE = HERE / "index.html"
SAMPLE_CSV = HERE / "sample_meter_readings.csv"
# Set by the Deploy workflow: the image tag (git commit) of the running version.
MODEL_VERSION = os.environ.get("MODEL_IMAGE_TAG", "local")
MAX_BODY = 20 * 1024 * 1024  # largest upload accepted (about a year of 15-minute readings)

# Median hourly import (kWh) per class in the RECON-SL training households: starting points for the form.
# Medians, not means: on a sunny day a solar home imports exactly zero at midday, which an
# average across many days blurs into a small positive value.
PRESETS = {
    "Typical day, home with solar": [0.429, 0.385, 0.361, 0.349, 0.359, 0.421, 0.312, 0.059, 0.001, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.047, 0.220, 0.424, 0.479, 0.497, 0.528, 0.540, 0.484],
    "Typical day, home without solar": [0.116, 0.111, 0.108, 0.107, 0.110, 0.120, 0.120, 0.100, 0.088, 0.087, 0.088,
                                        0.091, 0.096, 0.097, 0.099, 0.099, 0.099, 0.105, 0.132, 0.148, 0.152, 0.150,
                                        0.139, 0.123],
    "Empty house (fridge only)": [0.05] * 24,
}

# Plain-English names for the features the explanations mention most.
LABELS = {
    "total_kwh": "Total import in the day",
    "daytime_min_share": "Lowest daytime hour, as share of the day",
    "daytime_min_to_night_min": "Lowest daytime hour ÷ lowest night hour",
    "daytime_min_kwh": "Lowest daytime hour (kWh)",
    "daytime_zero_hours": "Daytime hours with ~zero import",
    "daytime_std": "Variation across daytime hours",
    "midday_kwh": "Import 10:00–15:00",
    "night_kwh": "Import 00:00–05:00",
    "evening_kwh": "Import 18:00–22:00",
    "night_min_kwh": "Lowest night hour (kWh)",
    "midday_to_night": "Midday ÷ night import",
    "midday_to_evening": "Midday ÷ evening import",
    "midday_to_morning": "Midday ÷ morning import",
    "bowl_depth": "Depth of the midday dip",
    "peak_hour": "Hour with the highest import",
    "trough_hour": "Hour with the lowest import",
}


def label(feature):
    """'h18_kwh' -> 'Import at 18:00 (kWh)'; known features get their plain-English name."""
    if feature in LABELS:
        return LABELS[feature]
    if feature.startswith("h") and feature[1:3].isdigit():
        hour, kind = feature[1:3], feature[4:]
        return f"Import at {hour}:00" + (" (share of day)" if kind == "share" else " (kWh)")
    return feature


class Model:
    """The trained detector plus its explainers. One copy, shared by every request."""

    def __init__(self):
        self.detector = joblib.load(MODEL_PATH)
        self.meta = json.loads(META_PATH.read_text())
        self.lock = threading.Lock()  # one prediction at a time keeps memory use flat
        # Permutation SHAP on the whole forest (its trees split on rotated axes, so tree SHAP would explain
        # mixtures of features), against a small background of training days saved with the model.
        from src.explain import shap_explainer, shap_values
        self.shap = shap_explainer(self.detector, self.detector.info["background"])
        self.shap_values = shap_values
        self.lime = None  # built on first use (takes a second)

    def info(self):
        test = self.meta["test_households"]
        return {
            "model": self.meta["model"], "version": MODEL_VERSION,
            "trained": self.meta.get("trained_at", "2026-10-20")[:10], "threshold": self.meta["threshold"],
            "accuracy": test["hh_accuracy"], "errors": test["hh_errors"], "n": test["hh_n"], "auc": test["hh_auc"],
            "precision": test["hh_precision"], "recall": test["hh_recall"],
            "params": self.meta["params"], "presets": PRESETS, "hours": HOURS,
        }

    def features(self, values):
        return self.detector.features(pd.DataFrame([values], columns=HOURS))

    def predict_day(self, values):
        X = self.features(values)
        with self.lock:
            p = float(self.detector.day_model.predict_proba(X)[:, 1][0])
            contrib = self.shap_values(self.shap, X).values[0]
        top = np.argsort(-np.abs(contrib))[:8]
        why = [{"feature": X.columns[i], "label": label(X.columns[i]), "value": round(float(X.iloc[0, i]), 4),
                "shap": round(float(contrib[i]), 4)} for i in top]
        return {"ok": True, "p": p, "solar": p >= self.meta["threshold"], "why": why}

    def explain_lime(self, values):
        X = self.features(values)
        with self.lock:
            if self.lime is None:
                from lime.lime_tabular import LimeTabularExplainer
                rng = np.random.default_rng(0)
                base = pd.DataFrame(list(PRESETS.values()), columns=HOURS)
                noise = pd.concat([base] * 200, ignore_index=True) * rng.lognormal(0, 0.5, (600, 24))
                bg = make_features(noise)[self.detector.feature_names]
                self.lime = LimeTabularExplainer(bg.values, feature_names=list(bg.columns),
                                                 class_names=["no solar", "solar"],
                                                 discretize_continuous=True, random_state=0)
            exp = self.lime.explain_instance(
                X.iloc[0].values,
                lambda rows: self.detector.day_model.predict_proba(pd.DataFrame(rows, columns=X.columns)),
                num_features=8, num_samples=2000)
        rules = [{"rule": rule, "weight": round(float(w), 4)} for rule, w in exp.as_list(label=1)]
        for r in rules:  # make the rules readable: replace feature names inside them
            for f in sorted(X.columns, key=len, reverse=True):
                if f in r["rule"]:
                    r["rule"] = r["rule"].replace(f, label(f))
                    break
        return {"ok": True, "rules": rules}

    def household(self, csv_text):
        readings = pd.read_csv(io.StringIO(csv_text))
        missing = {"timestamp", "import_kwh"} - set(readings.columns)
        if missing:
            return {"ok": False, "message": f"The CSV needs the column(s): {', '.join(sorted(missing))}."}
        days = readings_to_days(readings)
        if days.empty:
            return {"ok": False, "message": "No complete days found (a day needs at least 90 of 96 readings)."}
        with self.lock:
            days["p"] = self.detector.predict_day_proba(days)
        mean_p = float(days["p"].mean())
        return {
            "ok": True, "p": mean_p, "solar": mean_p >= self.meta["threshold"], "nDays": int(len(days)),
            "days": [{"day": str(d.date()), "p": round(float(p), 4)} for d, p in zip(days["day"], days["p"])],
            "profile": [round(float(v), 4) for v in days[HOURS].mean().values],
            "message": f"{len(days)} complete days from {days['day'].min().date()} to {days['day'].max().date()}.",
        }


def log_prediction(mode, p, n_days, verdict):
    """One JSON line per prediction on stdout -> Container Apps -> Log Analytics (monitoring).

    Only the score and counts are logged, never the readings themselves.
    """
    print(json.dumps({"event": "prediction", "mode": mode, "p_solar": round(p, 4), "n_days": n_days,
                      "verdict": int(verdict), "model_version": MODEL_VERSION,
                      "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)


def hours_from(body):
    """The 24 hourly values from the browser, cleaned: anything that is not a number >= 0 becomes 0."""
    values = list(body["hours"])
    if len(values) != 24:
        raise ValueError("need 24 values")
    out = []
    for v in values:
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        out.append(min(max(v, 0.0), 50.0) if math.isfinite(v) else 0.0)
    return out


def predict(body):
    r = model.predict_day(hours_from(body))
    log_prediction("day", r["p"], 1, r["solar"])
    return r


def household(body):
    r = model.household(str(body["csv"]))
    if r["ok"]:
        log_prediction("household", r["p"], r["nDays"], r["solar"])
    return r


# Every button in the page posts to one of these URLs; each gets the request body as a dict.
ROUTES = {
    "/api/predict": predict,
    "/api/lime": lambda body: model.explain_lime(hours_from(body)),
    "/api/household": household,
}

model = None  # created in main()


class Handler(BaseHTTPRequestHandler):
    """Handles each request from the browser."""

    def send(self, code, body, kind):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # "/"           -> the page itself (index.html)
        # "/health"     -> "ok" (used by Docker, the Deploy workflow and the daily Monitor check)
        # "/api/info"   -> model details, test results and the preset days, used when the page loads
        # "/api/sample" -> the bundled sample household's meter readings (CSV)
        path = self.path.split("?")[0]
        if path == "/":
            self.send(200, PAGE_FILE.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        elif path == "/health":
            self.send(200, "ok", "text/plain")
        elif path == "/api/info":
            self.send(200, json.dumps(model.info()), "application/json")
        elif path == "/api/sample":
            self.send(200, SAMPLE_CSV.read_text(encoding="utf-8"), "text/csv; charset=utf-8")
        else:
            self.send(404, "Not found", "text/plain")

    def do_POST(self):
        route = ROUTES.get(self.path)
        if not route:
            return self.send(404, "Not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            return self.send(413, json.dumps({"ok": False, "message": "File too large (max 20 MB)."}), "application/json")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            answer = route(body)
        except (ValueError, KeyError, TypeError, pd.errors.ParserError) as e:
            # Missing or broken data from the browser.
            answer = {"ok": False, "message": f"Could not read the input: {e}"}
        self.send(200, json.dumps(answer), "application/json")

    def log_message(self, *args):
        pass  # keep the terminal quiet: no line per request


def main():
    global model
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    print("Loading the model...", flush=True)
    model = Model()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{server.server_address[1]}/"
    print(f"Rooftop Solar Detector running at {url}  (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
