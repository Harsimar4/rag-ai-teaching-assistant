"""
Unit tests for rag_engine.py — the core RAG retrieval and generation logic.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rag_engine import (
    EmbeddingsNotFoundError,
    OllamaUnavailableError,
    RagEngineError,
    ask_question,
    build_prompt,
    check_health,
    chunks_to_sources,
    create_embedding,
    load_embeddings,
    retrieve_relevant_chunks,
    run_inference,
    seconds_to_mmss,
)


# ---------------------------------------------------------------------------
# seconds_to_mmss
# ---------------------------------------------------------------------------

class TestSecondsToMmss:
    def test_zero(self):
        assert seconds_to_mmss(0) == "0:00"

    def test_under_a_minute(self):
        assert seconds_to_mmss(45) == "0:45"

    def test_exact_minute(self):
        assert seconds_to_mmss(60) == "1:00"

    def test_mixed(self):
        assert seconds_to_mmss(125) == "2:05"

    def test_large(self):
        assert seconds_to_mmss(3661) == "61:01"

    def test_float_input(self):
        assert seconds_to_mmss(90.7) == "1:30"

    def test_single_digit_seconds_zero_padded(self):
        assert seconds_to_mmss(3) == "0:03"


# ---------------------------------------------------------------------------
# retrieve_relevant_chunks
# ---------------------------------------------------------------------------

class TestRetrieveRelevantChunks:
    @pytest.fixture
    def sample_df(self):
        """Create a small DataFrame with fake embeddings for testing."""
        data = [
            {
                "number": "1",
                "title": "Intro",
                "start": 0.0,
                "end": 30.0,
                "text": "Introduction to the course",
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "number": "1",
                "title": "Intro",
                "start": 30.0,
                "end": 60.0,
                "text": "Setting up your environment",
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "number": "2",
                "title": "Basics",
                "start": 0.0,
                "end": 45.0,
                "text": "Basic concepts explained",
                "embedding": [0.0, 0.0, 1.0],
            },
        ]
        return pd.DataFrame(data)

    def test_returns_top_k_results(self, sample_df):
        query_embedding = [1.0, 0.0, 0.0]  # most similar to first row
        results = retrieve_relevant_chunks(sample_df, query_embedding, top_k=2)
        assert len(results) == 2

    def test_most_similar_has_highest_score(self, sample_df):
        query_embedding = [1.0, 0.0, 0.0]
        results = retrieve_relevant_chunks(sample_df, query_embedding, top_k=3)
        scores = results["score"].tolist()
        # Highest score should be 1.0 (exact match with first row)
        assert max(scores) == pytest.approx(1.0)

    def test_results_sorted_by_start_time(self, sample_df):
        query_embedding = [0.5, 0.5, 0.0]
        results = retrieve_relevant_chunks(sample_df, query_embedding, top_k=3)
        starts = results["start"].tolist()
        assert starts == sorted(starts)

    def test_mmss_columns_added(self, sample_df):
        query_embedding = [1.0, 0.0, 0.0]
        results = retrieve_relevant_chunks(sample_df, query_embedding, top_k=1)
        assert "start_mmss" in results.columns
        assert "end_mmss" in results.columns

    def test_top_k_greater_than_data(self, sample_df):
        query_embedding = [1.0, 0.0, 0.0]
        results = retrieve_relevant_chunks(sample_df, query_embedding, top_k=10)
        assert len(results) == 3  # only 3 rows exist


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_query(self):
        df = pd.DataFrame(
            [{"title": "T", "number": "1", "start_mmss": "0:00", "end_mmss": "1:00", "text": "hello"}]
        )
        prompt = build_prompt("test question", df)
        assert "test question" in prompt

    def test_contains_instructions(self):
        df = pd.DataFrame(
            [{"title": "T", "number": "1", "start_mmss": "0:00", "end_mmss": "1:00", "text": "hello"}]
        )
        prompt = build_prompt("anything", df)
        assert "Answer like a human teacher" in prompt
        assert "EXACT timestamp" in prompt

    def test_is_course_agnostic(self):
        df = pd.DataFrame(
            [{"title": "T", "number": "1", "start_mmss": "0:00", "end_mmss": "1:00", "text": "hello"}]
        )
        prompt = build_prompt("anything", df)
        assert "Sigma" not in prompt
        assert "web development" not in prompt.lower()


# ---------------------------------------------------------------------------
# chunks_to_sources
# ---------------------------------------------------------------------------

class TestChunksToSources:
    def test_basic_conversion(self):
        df = pd.DataFrame(
            [
                {
                    "number": " 3 ",
                    "title": " My Title ",
                    "start_mmss": "1:00",
                    "end_mmss": "2:30",
                    "score": 0.87654,
                    "text": " excerpt text ",
                },
            ]
        )
        sources = chunks_to_sources(df)
        assert len(sources) == 1
        src = sources[0]
        assert src["video_number"] == "3"
        assert src["title"] == "My Title"
        assert src["start"] == "1:00"
        assert src["end"] == "2:30"
        assert src["score"] == 0.877
        assert src["excerpt"] == "excerpt text"

    def test_excerpt_truncated_to_200(self):
        df = pd.DataFrame(
            [
                {
                    "number": "1",
                    "title": "T",
                    "start_mmss": "0:00",
                    "end_mmss": "1:00",
                    "score": 0.5,
                    "text": "x" * 300,
                },
            ]
        )
        sources = chunks_to_sources(df)
        assert len(sources[0]["excerpt"]) == 200

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            columns=["number", "title", "start_mmss", "end_mmss", "score", "text"]
        )
        sources = chunks_to_sources(df)
        assert sources == []


# ---------------------------------------------------------------------------
# create_embedding (mocked)
# ---------------------------------------------------------------------------

class TestCreateEmbedding:
    @patch("rag_engine.requests.post")
    def test_returns_embeddings(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }
        mock_post.return_value = mock_response

        result = create_embedding(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("rag_engine.requests.post")
    def test_raises_on_connection_error(self, mock_post):
        import requests as req

        mock_post.side_effect = req.exceptions.ConnectionError("refused")
        with pytest.raises(OllamaUnavailableError):
            create_embedding(["hello"])

    @patch("rag_engine.requests.post")
    def test_raises_on_unexpected_response(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"error": "bad model"}
        mock_post.return_value = mock_response

        with pytest.raises(RagEngineError, match="Unexpected"):
            create_embedding(["hello"])


# ---------------------------------------------------------------------------
# run_inference (mocked)
# ---------------------------------------------------------------------------

class TestRunInference:
    @patch("rag_engine.requests.post")
    def test_returns_response_text(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "Here is the answer."}
        mock_post.return_value = mock_response

        result = run_inference("test prompt")
        assert result == "Here is the answer."

    @patch("rag_engine.requests.post")
    def test_raises_on_connection_error(self, mock_post):
        import requests as req

        mock_post.side_effect = req.exceptions.ConnectionError("refused")
        with pytest.raises(OllamaUnavailableError):
            run_inference("test prompt")


# ---------------------------------------------------------------------------
# load_embeddings (mocked)
# ---------------------------------------------------------------------------

class TestLoadEmbeddings:
    @patch("rag_engine.Path.exists", return_value=False)
    def test_raises_when_file_missing(self, _mock_exists):
        # Clear cache so it actually tries to load
        import rag_engine
        rag_engine._df_cache = None

        with pytest.raises(EmbeddingsNotFoundError, match="Embeddings file not found"):
            load_embeddings()

    @patch("rag_engine.joblib.load")
    @patch("rag_engine.Path.exists", return_value=True)
    def test_returns_dataframe(self, _mock_exists, mock_load):
        import rag_engine
        rag_engine._df_cache = None

        expected_df = pd.DataFrame({"col": [1, 2, 3]})
        mock_load.return_value = expected_df

        result = load_embeddings()
        pd.testing.assert_frame_equal(result, expected_df)

    @patch("rag_engine.joblib.load")
    @patch("rag_engine.Path.exists", return_value=True)
    def test_caching(self, _mock_exists, mock_load):
        import rag_engine
        rag_engine._df_cache = None

        expected_df = pd.DataFrame({"col": [1, 2]})
        mock_load.return_value = expected_df

        load_embeddings()
        load_embeddings()

        # joblib.load should only be called once due to caching
        mock_load.assert_called_once()

        # Clean up
        rag_engine._df_cache = None


# ---------------------------------------------------------------------------
# ask_question (integration, mocked external deps)
# ---------------------------------------------------------------------------

class TestAskQuestion:
    def test_empty_query_raises(self):
        with pytest.raises(RagEngineError, match="empty"):
            ask_question("")

    def test_whitespace_query_raises(self):
        with pytest.raises(RagEngineError, match="empty"):
            ask_question("   ")

    @patch("rag_engine.run_inference", return_value="Watch video 1 at 0:30")
    @patch("rag_engine.create_embedding", return_value=[[1.0, 0.0, 0.0]])
    @patch("rag_engine.load_embeddings")
    def test_full_pipeline(self, mock_load, mock_embed, mock_infer):
        import rag_engine
        rag_engine._df_cache = None

        mock_load.return_value = pd.DataFrame(
            [
                {
                    "number": "1",
                    "title": "Intro",
                    "start": 0.0,
                    "end": 30.0,
                    "text": "Introduction",
                    "embedding": [1.0, 0.0, 0.0],
                },
            ]
        )

        result = ask_question("What is covered?")
        assert result["question"] == "What is covered?"
        assert result["answer"] == "Watch video 1 at 0:30"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["video_number"] == "1"

        rag_engine._df_cache = None


# ---------------------------------------------------------------------------
# check_health (mocked)
# ---------------------------------------------------------------------------

class TestCheckHealth:
    @patch("rag_engine.requests.get")
    @patch("rag_engine.load_embeddings")
    def test_all_healthy(self, mock_load, mock_get):
        import rag_engine
        rag_engine._df_cache = None

        mock_load.return_value = pd.DataFrame({"a": [1, 2, 3]})
        mock_response = MagicMock()
        mock_response.ok = True
        mock_get.return_value = mock_response

        result = check_health()
        assert result["embeddings_loaded"] is True
        assert result["ollama_reachable"] is True
        assert result["ready"] is True
        assert result["chunk_count"] == 3

        rag_engine._df_cache = None

    @patch("rag_engine.requests.get")
    @patch("rag_engine.load_embeddings", side_effect=EmbeddingsNotFoundError("missing"))
    def test_embeddings_missing(self, mock_load, mock_get):
        import rag_engine
        rag_engine._df_cache = None

        mock_response = MagicMock()
        mock_response.ok = True
        mock_get.return_value = mock_response

        result = check_health()
        assert result["embeddings_loaded"] is False
        assert result["ollama_reachable"] is True
        assert result["ready"] is False

        rag_engine._df_cache = None

    @patch("rag_engine.requests.get")
    @patch("rag_engine.load_embeddings")
    def test_ollama_unreachable(self, mock_load, mock_get):
        import rag_engine
        rag_engine._df_cache = None
        import requests as req

        mock_load.return_value = pd.DataFrame({"a": [1]})
        mock_get.side_effect = req.exceptions.ConnectionError("refused")

        result = check_health()
        assert result["embeddings_loaded"] is True
        assert result["ollama_reachable"] is False
        assert result["ready"] is False

        rag_engine._df_cache = None
