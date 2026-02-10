"""Tests for FastAPI API endpoints."""

import pytest


# ── POST /api/query ───────────────────────────────────────────────────


@pytest.mark.api
class TestQueryEndpoint:

    def test_query_returns_answer_and_session(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("RAG answer", [])

        resp = client.post("/api/query", json={"query": "What is RAG?"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "RAG answer"
        assert body["session_id"] == "session_1"
        assert body["sources"] == []

    def test_query_creates_session_when_none_provided(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "hello"})
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, client, mock_rag_system):
        resp = client.post(
            "/api/query", json={"query": "hello", "session_id": "my_session"}
        )
        body = resp.json()
        assert body["session_id"] == "my_session"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("hello", "my_session")

    def test_query_returns_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "answer",
            [{"text": "RAG - Lesson 1", "url": "https://example.com"}],
        )

        resp = client.post("/api/query", json={"query": "q"})

        body = resp.json()
        assert len(body["sources"]) == 1
        assert body["sources"][0]["text"] == "RAG - Lesson 1"
        assert body["sources"][0]["url"] == "https://example.com"

    def test_query_source_with_null_url(self, client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "answer",
            [{"text": "Course A", "url": None}],
        )

        resp = client.post("/api/query", json={"query": "q"})

        assert resp.status_code == 200
        assert resp.json()["sources"][0]["url"] is None

    def test_query_returns_500_on_rag_error(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB connection lost")

        resp = client.post("/api/query", json={"query": "q"})

        assert resp.status_code == 500
        assert "DB connection lost" in resp.json()["detail"]

    def test_query_missing_body_returns_422(self, client):
        resp = client.post("/api/query")
        assert resp.status_code == 422

    def test_query_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422

    def test_query_empty_string_is_valid(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": ""})
        assert resp.status_code == 200


# ── GET /api/courses ──────────────────────────────────────────────────


@pytest.mark.api
class TestCoursesEndpoint:

    def test_courses_returns_stats(self, client, mock_rag_system):
        resp = client.get("/api/courses")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 2
        assert body["course_titles"] == ["Course A", "Course B"]

    def test_courses_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        resp = client.get("/api/courses")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []

    def test_courses_returns_500_on_error(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("fail")

        resp = client.get("/api/courses")

        assert resp.status_code == 500
        assert "fail" in resp.json()["detail"]


# ── DELETE /api/session/{session_id} ──────────────────────────────────


@pytest.mark.api
class TestDeleteSessionEndpoint:

    def test_delete_existing_session(self, client, mock_rag_system):
        mock_rag_system.session_manager.sessions["s1"] = []

        resp = client.delete("/api/session/s1")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert "s1" not in mock_rag_system.session_manager.sessions

    def test_delete_nonexistent_session_is_ok(self, client):
        resp = client.delete("/api/session/does_not_exist")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
