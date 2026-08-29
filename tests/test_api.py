"""
Unit tests for the FastAPI server (api.py).
"""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_root_returns_message(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "RAG Course Assistant API" in data["message"]
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"

    def test_root_has_no_sigma_branding(self):
        resp = client.get("/")
        data = resp.json()
        assert "Sigma" not in data["message"]


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    @patch("api.check_health")
    def test_health_online(self, mock_health):
        mock_health.return_value = {
            "embeddings_loaded": True,
            "ollama_reachable": True,
            "chunk_count": 42,
            "ready": True,
        }
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["chunk_count"] == 42

    @patch("api.check_health")
    def test_health_offline(self, mock_health):
        mock_health.return_value = {
            "embeddings_loaded": False,
            "ollama_reachable": False,
            "chunk_count": 0,
            "ready": False,
        }
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is False


# ---------------------------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    @patch("api.ask_question")
    def test_ask_success(self, mock_ask):
        mock_ask.return_value = {
            "question": "How to start?",
            "answer": "Watch video 1 at 0:00.",
            "sources": [
                {
                    "video_number": "1",
                    "title": "Getting Started",
                    "start": "0:00",
                    "end": "5:00",
                    "score": 0.95,
                    "excerpt": "In this video we learn...",
                }
            ],
        }

        resp = client.post("/ask", json={"question": "How to start?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "How to start?"
        assert data["answer"] == "Watch video 1 at 0:00."
        assert len(data["sources"]) == 1

    def test_ask_empty_question(self):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422  # validation error

    def test_ask_missing_question(self):
        resp = client.post("/ask", json={})
        assert resp.status_code == 422

    @patch("api.ask_question")
    def test_ask_question_too_long(self, mock_ask):
        resp = client.post("/ask", json={"question": "x" * 501})
        assert resp.status_code == 422

    @patch("api.ask_question")
    def test_ask_embeddings_not_found(self, mock_ask):
        from rag_engine import EmbeddingsNotFoundError

        mock_ask.side_effect = EmbeddingsNotFoundError("embeddings missing")
        resp = client.post("/ask", json={"question": "test"})
        assert resp.status_code == 503
        assert "embeddings missing" in resp.json()["detail"]

    @patch("api.ask_question")
    def test_ask_ollama_unavailable(self, mock_ask):
        from rag_engine import OllamaUnavailableError

        mock_ask.side_effect = OllamaUnavailableError("Ollama down")
        resp = client.post("/ask", json={"question": "test"})
        assert resp.status_code == 503
        assert "Ollama down" in resp.json()["detail"]

    @patch("api.ask_question")
    def test_ask_generic_error(self, mock_ask):
        mock_ask.side_effect = RuntimeError("unexpected")
        resp = client.post("/ask", json={"question": "test"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error."

    @patch("api.ask_question")
    def test_ask_rag_engine_error(self, mock_ask):
        from rag_engine import RagEngineError

        mock_ask.side_effect = RagEngineError("bad query")
        resp = client.post("/ask", json={"question": "test"})
        assert resp.status_code == 400
        assert "bad query" in resp.json()["detail"]
