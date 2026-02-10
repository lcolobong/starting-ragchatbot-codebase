"""Tests for RAGSystem.query() pipeline orchestration."""

from unittest.mock import patch, MagicMock, PropertyMock
from rag_system import RAGSystem


def _build_rag_system():
    """Construct a RAGSystem with all heavy dependencies mocked out."""
    with (
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.VectorStore"),
        patch("rag_system.AIGenerator") as MockGen,
        patch("rag_system.SessionManager") as MockSession,
        patch("rag_system.CourseSearchTool"),
        patch("rag_system.CourseOutlineTool"),
        patch("rag_system.ToolManager") as MockTM,
    ):

        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_db"
        config.EMBEDDING_MODEL = "test"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "fake"
        config.ANTHROPIC_MODEL = "m"
        config.MAX_HISTORY = 2

        system = RAGSystem(config)
    return system


# ── Prompt template ─────────────────────────────────────────────────────


def test_query_wraps_in_prompt_template():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = []

    system.query("What is RAG?")

    call_kwargs = system.ai_generator.generate_response.call_args[1]
    assert (
        call_kwargs["query"]
        == "Answer this question about course materials: What is RAG?"
    )


# ── Session history ─────────────────────────────────────────────────────


def test_query_retrieves_session_history():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = []
    system.session_manager.get_conversation_history.return_value = "prev"

    system.query("q", session_id="s1")

    system.session_manager.get_conversation_history.assert_called_once_with("s1")


def test_query_passes_history_to_generator():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = []
    system.session_manager.get_conversation_history.return_value = "User: hi\nAI: hello"

    system.query("q", session_id="s1")

    call_kwargs = system.ai_generator.generate_response.call_args[1]
    assert call_kwargs["conversation_history"] == "User: hi\nAI: hello"


def test_query_no_session_skips_history():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = []

    system.query("q", session_id=None)

    system.session_manager.get_conversation_history.assert_not_called()


# ── Tools and tool_manager forwarding ───────────────────────────────────


def test_query_passes_tools_and_manager():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = []
    system.tool_manager.get_tool_definitions.return_value = [{"name": "search"}]

    system.query("q")

    call_kwargs = system.ai_generator.generate_response.call_args[1]
    assert call_kwargs["tools"] == [{"name": "search"}]
    assert call_kwargs["tool_manager"] is system.tool_manager


# ── Sources ─────────────────────────────────────────────────────────────


def test_query_collects_sources():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = [
        {"text": "RAG - Lesson 1", "url": "https://example.com"}
    ]

    _, sources = system.query("q")

    assert sources == [{"text": "RAG - Lesson 1", "url": "https://example.com"}]


def test_query_resets_sources_after_retrieval():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = []

    system.query("q")

    system.tool_manager.reset_sources.assert_called_once()


# ── Session updates ─────────────────────────────────────────────────────


def test_query_updates_session_with_exchange():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "the answer"
    system.tool_manager.get_last_sources.return_value = []

    system.query("my question", session_id="s1")

    system.session_manager.add_exchange.assert_called_once_with(
        "s1", "my question", "the answer"
    )


def test_query_skips_session_update_without_id():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = []

    system.query("q", session_id=None)

    system.session_manager.add_exchange.assert_not_called()


# ── Return structure ────────────────────────────────────────────────────


def test_query_returns_correct_tuple_structure():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = [{"text": "src"}]

    result = system.query("q")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)


def test_query_empty_sources_when_no_tool_used():
    system = _build_rag_system()
    system.ai_generator.generate_response.return_value = "answer"
    system.tool_manager.get_last_sources.return_value = []

    _, sources = system.query("q")

    assert sources == []
