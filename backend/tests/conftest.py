import pytest
from unittest.mock import MagicMock, Mock, patch
from vector_store import VectorStore, SearchResults


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
