"""
Integration Tests for FastAPI Backend Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "supported_models" in data
    assert len(data["supported_models"]) == 3


def test_get_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) == 3
    keys = [m["model_key"] for m in models]
    assert "bart_base_dense" in keys
    assert "led_base_sparse_4k" in keys
    assert "qwen25_1_5b_causal_qlora" in keys


def test_get_pareto_benchmarks_endpoint(client):
    response = client.get("/api/benchmarks/pareto")
    assert response.status_code == 200
    data = response.json()
    assert data["total_models"] >= 3
    assert len(data["benchmarks"]) >= 3


def test_extract_entities_endpoint(client):
    req = {
        "text": "Under Section 104(a) and 42 U.S.C. § 1983, $25,000,000 is authorized with a 15% set-aside."
    }
    response = client.post("/api/extract-entities", json=req)
    assert response.status_code == 200
    data = response.json()
    assert len(data["sections"]) > 0
    assert len(data["usc_citations"]) > 0
    assert len(data["monetary_amounts"]) > 0
    assert len(data["percentages"]) > 0
    assert data["total_count"] >= 4


def test_summarize_single_model_dry_run(client):
    req = {
        "text": "SECTION 1. Short title. The Infrastructure Act authorizes appropriations for water treatment.",
        "model_key": "bart_base_dense",
        "dry_run": True,
    }
    response = client.post("/api/summarize", json=req)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    res = data["results"][0]
    assert res["model_key"] == "bart_base_dense"
    assert len(res["summary_text"]) > 0
    assert res["latency_ms"] > 0


def test_summarize_all_models_dry_run(client):
    req = {
        "text": "SECTION 1. Veterans mental health care expansion and facility grants.",
        "model_key": "all",
        "dry_run": True,
    }
    response = client.post("/api/summarize", json=req)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    keys = [r["model_key"] for r in data["results"]]
    assert "bart_base_dense" in keys
    assert "led_base_sparse_4k" in keys
    assert "qwen25_1_5b_causal_qlora" in keys
    assert data["fastest_model"] is not None


def test_summarize_selected_models_dry_run(client):
    req = {
        "text": "SECTION 1. This bill establishes a water infrastructure grant program.",
        "models": ["led", "bart_base_dense", "led_base_sparse_4k"],
        "dry_run": True,
    }
    response = client.post("/api/summarize", json=req)

    assert response.status_code == 200
    results = response.json()["results"]
    assert [result["model_key"] for result in results] == [
        "led_base_sparse_4k",
        "bart_base_dense",
    ]


def test_summarize_rejects_empty_models_list(client):
    response = client.post(
        "/api/summarize",
        json={"text": "A bill.", "models": [], "dry_run": True},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Select at least one model."


def test_evaluate_endpoint(client):
    req = {
        "candidate_summary": "Pursuant to 33 U.S.C. 1251, Section 104 authorizes $450,000,000.",
        "reference_summary": "Under 33 U.S.C. 1251, Section 104 provides $450,000,000 for water infrastructure.",
        "use_bertscore": False,
    }
    response = client.post("/api/evaluate", json=req)
    assert response.status_code == 200
    data = response.json()
    assert data["rouge1"] > 0
    assert data["legal_entity_recall_pct"] > 0
    assert len(data["matched_entities"]) > 0
