"""End-to-end tests of the web app: start app/server.py on a free port and call its URLs."""
import importlib.util
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "solar_detector.joblib").exists(),
                                reason="trained model not available")


@pytest.fixture(scope="module")
def base_url():
    spec = importlib.util.spec_from_file_location("server", ROOT / "app" / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
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
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_page_health_and_info(base_url):
    url, _ = base_url
    assert get(url + "/health") == (200, "ok")
    status, page = get(url + "/")
    assert status == 200 and "Rooftop" in page
    info = json.loads(get(url + "/api/info")[1])
    assert info["n"] > 0 and 0 < info["threshold"] < 1 and len(info["hours"]) == 24


def test_predict_solar_and_non_solar_days(base_url):
    url, server = base_url
    solar = post(url + "/api/predict", {"hours": server.PRESETS["Typical day, home with solar"]})
    flat = post(url + "/api/predict", {"hours": server.PRESETS["Typical day, home without solar"]})
    assert solar["ok"] and solar["solar"] and not flat["solar"]
    assert len(solar["why"]) == 8 and all("label" in w and "shap" in w for w in solar["why"])


def test_bad_input_is_rejected_politely(base_url):
    url, _ = base_url
    r = post(url + "/api/predict", {"hours": [1, 2, 3]})
    assert r["ok"] is False and "input" in r["message"]


def test_household_upload_with_sample(base_url):
    url, _ = base_url
    csv = get(url + "/api/sample")[1]
    r = post(url + "/api/household", {"csv": csv})
    assert r["ok"] and r["solar"] and r["nDays"] >= 20 and len(r["profile"]) == 24


def test_household_upload_with_wrong_columns(base_url):
    url, _ = base_url
    r = post(url + "/api/household", {"csv": "a,b\n1,2\n"})
    assert r["ok"] is False and "timestamp" in r["message"]
