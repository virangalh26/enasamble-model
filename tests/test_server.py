"""End-to-end tests of the web app: start app/server.py on a free port (with the fixture's model) and call it."""
import importlib.util
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base_url(pipeline):
    spec = importlib.util.spec_from_file_location("server", ROOT / "app" / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    server.MODEL_PATH = pipeline["models"] / "poverty_targeter.joblib"
    server.META_PATH = pipeline["models"] / "metadata.json"
    server.DEFAULTS_PATH = pipeline["models"] / "form_defaults.json"
    server.model = server.Model()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", server
    httpd.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.status, r.read().decode()


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_page_health_and_info(base_url):
    url, _ = base_url
    assert get(url + "/health") == (200, "ok")
    status, page = get(url + "/")
    assert status == 200 and "Poverty" in page
    info = json.loads(get(url + "/api/info")[1])
    assert info["povertyLine"] == 6966 and 0 < info["threshold"] < 1
    names = {f["name"] for f in info["fields"]}
    assert {"district", "walls", "own_fridge", "n_members"} <= names
    district = next(f for f in info["fields"] if f["name"] == "district")
    assert all(not o["label"].startswith("code") for o in district["options"])


def test_poorer_visit_scores_higher(base_url):
    url, _ = base_url
    rich = {"n_members": 3, "own_fridge": 1, "own_washing_mechine": 1, "own_motor_car_van": 1, "own_computers": 1,
            "bed_rooms": 4, "area": 6, "head_education": 15, "max_adult_education": 17}
    poor = {"n_members": 7, "n_children": 4, "own_fridge": 0, "own_washing_mechine": 0, "own_motor_car_van": 0,
            "own_computers": 0, "own_tv": 0, "bed_rooms": 1, "area": 1, "head_education": 2, "max_adult_education": 4}
    a, b = post(url + "/api/predict", {"values": rich}), post(url + "/api/predict", {"values": poor})
    assert a["ok"] and b["ok"] and b["p"] > a["p"]
    assert a["low"] < a["pc"] < a["high"] and len(a["why"]) > 0


def test_lime_returns_rules(base_url):
    url, _ = base_url
    r = post(url + "/api/lime", {"values": {"n_members": 5}})
    assert r["ok"] and len(r["rules"]) == 8


def test_bad_input_is_rejected_politely(base_url):
    url, _ = base_url
    r = post(url + "/api/predict", {"values": {"n_members": "many"}})
    assert r["ok"] is False and "n_members" in r["message"]
