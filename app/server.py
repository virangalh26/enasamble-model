"""Aswesuma-style Poverty Screener: a small web app made of this Python server and one HTML page.

A welfare officer fills in what they can see or ask on a home visit; the NGBoost model returns the chance that
the household is below the official poverty line, the predicted spending per person with an 80% range, and the
reasons (SHAP and LIME). index.html, next to it, is the whole screen: it calls the /api/... URLs below.
It uses only the Python standard library for the web part (no Streamlit or Flask).

Run on this computer (opens the browser):
    python app/server.py
Run in the container on Azure:
    python app/server.py --host 0.0.0.0 --port 8501 --no-browser
"""
import argparse
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
from src.features import CATEGORICAL, DURABLES  # noqa: E402
from src.hies import DISTRICTS, SECTORS  # noqa: E402

MODEL_PATH = ROOT / "models" / "poverty_targeter.joblib"
META_PATH = ROOT / "models" / "metadata.json"
DEFAULTS_PATH = ROOT / "models" / "form_defaults.json"
CODE_LABELS_PATH = HERE / "code_labels.json"
PAGE_FILE = HERE / "index.html"
# Set by the Deploy workflow: the image tag (git commit) of the running version.
MODEL_VERSION = os.environ.get("MODEL_IMAGE_TAG", "local")
MAX_BODY = 1024 * 1024

# What the officer is asked, in the order shown on the form. Everything else keeps the typical value of
# training households (models/form_defaults.json), so a short form still gives a full model input.
FORM = [
    ("Location", ["district", "sector"]),
    ("People", ["n_members", "n_children", "n_elderly", "n_employed", "head_female", "head_age", "head_education",
                "max_adult_education"]),
    ("House", ["walls", "floor", "roof", "bed_rooms", "area", "ownership", "toilet_type", "drinking_water",
               "cooking_fuel", "lite_source"]),
    ("Belongings", ["own_" + d for d in ["fridge", "tv", "washing_mechine", "computers", "motor_bicycle",
                                         "three_wheeler", "motor_car_van", "telephone_mobile", "s6a_aircon",
                                         "sewingmechine"]]),
    ("Land & animals", ["owns_agri_land", "land_perches", "has_s9_cattle_buffaloes", "has_chickens"]),
]

LABELS = {
    "district": "District", "sector": "Sector", "n_members": "People in the household",
    "n_children": "Children under 15", "n_elderly": "People aged 60+", "n_employed": "People with work",
    "head_female": "Head of household is a woman", "head_age": "Age of the head",
    "head_education": "Education of the head (grade code)", "max_adult_education": "Highest adult education (code)",
    "walls": "Wall material", "floor": "Floor material", "roof": "Roof material", "bed_rooms": "Bedrooms",
    "area": "Floor area (size class)", "ownership": "Ownership of the house", "toilet_type": "Toilet type",
    "drinking_water": "Main drinking water", "cooking_fuel": "Cooking fuel", "lite_source": "Lighting source",
    "owns_agri_land": "Owns agricultural land", "land_perches": "Land owned (perches)",
    "has_s9_cattle_buffaloes": "Has cattle or buffaloes", "has_chickens": "Has chickens",
    "n_durables": "Number of durable goods owned", "dependency_ratio": "Dependants per working-age adult",
    "employed_share": "Share of members with work", "share_female": "Share of members who are women",
    "n_working_age": "People aged 15-59", "mean_adult_education": "Average adult education (code)",
    **{"own_" + d: "Owns " + d.replace("s6a_", "").replace("_", " ").replace("mechine", "machine") for d in DURABLES},
}
BINARY = {"head_female", "owns_agri_land", "has_s9_cattle_buffaloes", "has_chickens"} | {"own_" + d for d in DURABLES}


def code_labels():
    """Names of HIES answer codes (e.g. walls: 1 = brick). Unknown codes are shown as 'code k'."""
    labels = json.loads(CODE_LABELS_PATH.read_text()) if CODE_LABELS_PATH.exists() else {}
    labels["district"] = {str(k): v for k, v in DISTRICTS.items()}
    labels["sector"] = {str(k): v for k, v in SECTORS.items()}
    return labels


def label(feature):
    base = feature.split("=")[0]
    name = LABELS.get(base, base.replace("_", " ").capitalize())
    if "=" in feature:
        code = feature.split("=")[1]
        name += ": " + CODES.get(base, {}).get(code, f"code {code}")
    return name


CODES = code_labels()


class Model:
    def __init__(self):
        import shap
        self.t = joblib.load(MODEL_PATH)
        self.meta = json.loads(META_PATH.read_text())
        self.defaults = json.loads(DEFAULTS_PATH.read_text())
        self.shap = shap.TreeExplainer(self.t.model, model_output=0)
        self.lime = None

    def info(self):
        test = self.meta["test"]
        fields = []
        for section, names in FORM:
            for f in names:
                d = self.defaults["values"].get(f, 0)
                if f in CATEGORICAL:
                    opts = [{"value": c, "label": CODES.get(f, {}).get(str(int(c)), f"code {int(c)}")}
                            for c in self.defaults["categories"].get(f, [])]
                    fields.append({"name": f, "label": label(f), "section": section, "type": "select",
                                   "options": opts, "default": d})
                else:
                    fields.append({"name": f, "label": label(f), "section": section,
                                   "type": "yesno" if f in BINARY else "number", "default": d})
        return {"model": self.meta["model"], "povertyLine": self.meta["poverty_line"],
                "threshold": self.meta["threshold"], "test": test, "regression": self.meta["test_regression"],
                "baselines": self.meta["baselines"], "fields": fields, "version": MODEL_VERSION,
                "nTrain": self.t.info.get("trained_households")}

    def household(self, values: dict) -> pd.DataFrame:
        row = dict(self.defaults["values"])
        for k, v in values.items():
            if k not in row:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{k} must be a number")
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{k} must be a number >= 0")
            row[k] = v
        # keep derived counts consistent with what the officer typed
        row["n_durables"] = sum(row.get("own_" + d, 0) for d in DURABLES)
        row["n_working_age"] = max(row["n_members"] - row["n_children"] - row["n_elderly"], 0)
        row["dependency_ratio"] = (row["n_children"] + row["n_elderly"]) / max(row["n_working_age"], 1)
        row["employed_share"] = row["n_employed"] / max(row["n_members"], 1)
        return pd.DataFrame([row])

    def predict(self, values: dict):
        hh = self.household(values)
        r = self.t.predict(hh).iloc[0]
        X = self.t.matrix(hh)
        sv = pd.Series(self.shap.shap_values(X.values)[0], index=X.columns)
        sv = sv[sv.abs() > 1e-6]
        top = sv.reindex(sv.abs().sort_values(ascending=False).index[:8])
        why = [{"label": label(f), "score": float(-v)} for f, v in top.items()]  # + = towards poverty
        return {"ok": True, "p": float(r.p_poor), "flag": bool(r.flag_for_support),
                "pc": float(r.pred_pc_consumption), "low": float(r.pc_low), "high": float(r.pc_high),
                "why": why}

    def explain_lime(self, values: dict):
        from lime.lime_tabular import LimeTabularExplainer
        if self.lime is None:
            sample = np.array(self.defaults["lime_background"])
            self.lime = LimeTabularExplainer(sample, feature_names=self.t.columns, mode="classification",
                                             class_names=["not poor", "poor"], random_state=0)
        X = self.t.matrix(self.household(values))
        f = lambda x: np.c_[1 - (q := self.t.p_poor_from_matrix(pd.DataFrame(x, columns=self.t.columns))), q]
        exp = self.lime.explain_instance(X.values[0], f, num_features=8, num_samples=2000)
        return {"ok": True, "rules": [{"label": rule, "score": float(w)} for rule, w in exp.as_list()]}


def log_prediction(p, flag):
    """One JSON line per prediction on stdout -> Container Apps -> Log Analytics (monitoring).

    Only the score and the decision are logged, never the household's answers.
    """
    print(json.dumps({"event": "prediction", "p_poor": round(p, 4), "flag": int(flag),
                      "model_version": MODEL_VERSION, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}),
          flush=True)


def predict(body):
    r = model.predict(dict(body.get("values", {})))
    log_prediction(r["p"], r["flag"])
    return r


ROUTES = {
    "/api/predict": predict,
    "/api/lime": lambda body: model.explain_lime(dict(body.get("values", {}))),
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
        # "/" -> the page · "/health" -> "ok" (Docker, Deploy, Monitor) · "/api/info" -> model details and the form
        path = self.path.split("?")[0]
        if path == "/":
            self.send(200, PAGE_FILE.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        elif path == "/health":
            self.send(200, "ok", "text/plain")
        elif path == "/api/info":
            self.send(200, json.dumps(model.info()), "application/json")
        else:
            self.send(404, "Not found", "text/plain")

    def do_POST(self):
        route = ROUTES.get(self.path)
        if not route:
            return self.send(404, "Not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            return self.send(413, json.dumps({"ok": False, "message": "Request too large."}), "application/json")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            answer = route(body)
        except (ValueError, KeyError, TypeError) as e:
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
    print(f"Poverty Screener running at {url}  (Ctrl+C to stop)", flush=True)
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
