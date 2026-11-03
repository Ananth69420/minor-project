"""
Automated tests for FastAPI Server REST & WebSocket endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)
FLEET_RL_DIR = os.path.join(ROOT_DIR, "fleet-rl")
if FLEET_RL_DIR not in sys.path:
    sys.path.insert(0, FLEET_RL_DIR)

from server import app, engine


@pytest.fixture
def client():
    return TestClient(app)


def test_api_status(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["grid_size"] == 16
    assert data["num_vehicles"] == 8


def test_api_state(client):
    res = client.get("/api/state")
    assert res.status_code == 200
    data = res.json()
    assert "vehicles" in data
    assert len(data["vehicles"]) == 8
    assert "metrics" in data
    assert "inspector" in data


def test_api_step(client):
    # Reset first
    client.post("/api/reset", json={"seed": 5000, "surge_quad": 0, "dispatcher": "rl", "mode": "inference"})
    res = client.post("/api/step")
    assert res.status_code == 200
    data = res.json()
    assert data["step"] == 1
    assert "step_info" in data


def test_api_dispatcher_and_mode(client):
    res = client.post("/api/dispatcher", json={"dispatcher": "baseline"})
    assert res.status_code == 200
    assert res.json()["dispatcher_type"] == "baseline"

    res = client.post("/api/mode", json={"mode": "training"})
    assert res.status_code == 200
    assert res.json()["mode"] == "training"

    # Reset back to rl and inference
    client.post("/api/dispatcher", json={"dispatcher": "rl"})
    client.post("/api/mode", json={"mode": "inference"})


def test_api_qtable(client):
    res = client.get("/api/qtable/198")
    assert res.status_code == 200
    data = res.json()
    assert data["state"]["state_id"] == 198
    assert len(data["actions"]) == 6


def test_api_compare(client):
    res = client.post("/api/compare", json={"seed": 5000, "surge_quad": 0})
    assert res.status_code == 200
    data = res.json()
    assert "baseline" in data
    assert "forecast_rule" in data
    assert "qlearning" in data
    assert "deltas_rl_vs_baseline" in data

