import pytest
from unittest.mock import MagicMock, Mock, patch
from vector_store import VectorStore, SearchResults


# ── Unit-test fixtures ────────────────────────────────────────────────


@pytest.fixture
def mock_vector_store():
    """A MagicMock that respects VectorStore's interface."""
    store = MagicMock(spec=VectorStore)
    store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    store.get_lesson_link.return_value = None
    store.get_course_link.return_value = None
    return store


@pytest.fixture
def search_results_factory():
    """Factory for creating SearchResults with test data."""
    def _make(documents=None, metadata=None, distances=None, error=None):
        return SearchResults(
            documents=documents or [],
            metadata=metadata or [],
            distances=distances or [],
            error=error,
        )
    return _make


def _make_text_block(text):
    """Build a mock content block with type='text'."""
    block = Mock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id, name, tool_input):
    """Build a mock content block with type='tool_use'."""
    block = Mock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input
    return block


@pytest.fixture
def mock_anthropic_client():
    """A mock Anthropic client whose messages.create() can be configured per-test."""
    client = MagicMock()
    return client


@pytest.fixture
def make_api_response():
    """Factory for creating mock Anthropic API responses."""
    def _make(content_blocks, stop_reason="end_turn"):
        resp = Mock()
        resp.content = content_blocks
        resp.stop_reason = stop_reason
        return resp
    return _make


# ── API-test fixtures ─────────────────────────────────────────────────


@pytest.fixture
def mock_rag_system():
    """A MagicMock standing in for RAGSystem with sensible defaults."""
    rag = MagicMock()
    rag.query.return_value = ("Test answer", [])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    rag.session_manager.create_session.return_value = "session_1"
    rag.session_manager.sessions = {}
    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """A FastAPI app with API routes but *no* static-file mount.

    This avoids the FileNotFoundError that importing backend/app.py
    would cause (it mounts ../frontend which doesn't exist in CI).
    The routes are thin wrappers so duplicating them here is fine;
    they are the contract under test.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    rag_system = mock_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        rag_system.session_manager.sessions.pop(session_id, None)
        return {"status": "ok"}

    return app


@pytest.fixture
def client(test_app):
    """Synchronous test client for the test app."""
    from starlette.testclient import TestClient

    with TestClient(test_app) as c:
        yield c
