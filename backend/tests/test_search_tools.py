"""Tests for CourseSearchTool.execute() output formatting and source tracking."""

from search_tools import CourseSearchTool
from vector_store import SearchResults

# ── Error handling ──────────────────────────────────────────────────────


def test_execute_returns_error_when_search_fails(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[], error="Connection timeout"
    )
    tool = CourseSearchTool(mock_vector_store)
    assert tool.execute(query="anything") == "Connection timeout"


# ── Empty results ───────────────────────────────────────────────────────


def test_execute_empty_results_no_filters(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)
    assert tool.execute(query="xyz") == "No relevant content found."


def test_execute_empty_results_with_course_filter(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="xyz", course_name="MCP")
    assert result == "No relevant content found in course 'MCP'."


def test_execute_empty_results_with_lesson_filter(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="xyz", lesson_number=5)
    assert result == "No relevant content found in lesson 5."


def test_execute_empty_results_with_both_filters(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="xyz", course_name="MCP", lesson_number=3)
    assert result == "No relevant content found in course 'MCP' in lesson 3."


def test_execute_empty_results_lesson_zero(mock_vector_store):
    """Bug detector: lesson_number=0 is falsy but valid.
    Line 82 uses `if lesson_number:` instead of `if lesson_number is not None:`.
    """
    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="intro", lesson_number=0)
    assert result == "No relevant content found in lesson 0."


# ── Formatted results ──────────────────────────────────────────────────


def test_execute_formats_single_result_with_lesson(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["Some content here"],
        metadata=[{"course_title": "RAG Course", "lesson_number": 2}],
        distances=[0.3],
    )
    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="rag")
    assert "[RAG Course - Lesson 2]" in result
    assert "Some content here" in result


def test_execute_formats_single_result_without_lesson(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["Overview text"],
        metadata=[{"course_title": "RAG Course"}],
        distances=[0.2],
    )
    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="overview")
    assert "[RAG Course]" in result
    assert "Lesson" not in result


def test_execute_formats_multiple_results(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["First chunk", "Second chunk"],
        metadata=[
            {"course_title": "A", "lesson_number": 1},
            {"course_title": "B", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )
    tool = CourseSearchTool(mock_vector_store)
    result = tool.execute(query="q")
    parts = result.split("\n\n")
    assert len(parts) == 2


# ── Source deduplication and URL resolution ─────────────────────────────


def test_execute_deduplicates_sources(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["chunk1", "chunk2"],
        metadata=[
            {"course_title": "RAG", "lesson_number": 1},
            {"course_title": "RAG", "lesson_number": 1},
        ],
        distances=[0.1, 0.2],
    )
    tool = CourseSearchTool(mock_vector_store)
    tool.execute(query="q")
    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["text"] == "RAG - Lesson 1"


def test_execute_url_prefers_lesson_link(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["doc"],
        metadata=[{"course_title": "C", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_vector_store.get_lesson_link.return_value = "https://lesson.url"
    tool = CourseSearchTool(mock_vector_store)
    tool.execute(query="q")
    assert tool.last_sources[0]["url"] == "https://lesson.url"
    mock_vector_store.get_course_link.assert_not_called()


def test_execute_url_falls_back_to_course_link(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["doc"],
        metadata=[{"course_title": "C", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_vector_store.get_lesson_link.return_value = None
    mock_vector_store.get_course_link.return_value = "https://course.url"
    tool = CourseSearchTool(mock_vector_store)
    tool.execute(query="q")
    assert tool.last_sources[0]["url"] == "https://course.url"


def test_execute_url_both_none(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["doc"],
        metadata=[{"course_title": "C", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_vector_store.get_lesson_link.return_value = None
    mock_vector_store.get_course_link.return_value = None
    tool = CourseSearchTool(mock_vector_store)
    tool.execute(query="q")
    assert tool.last_sources[0]["url"] is None


# ── Parameter pass-through ──────────────────────────────────────────────


def test_execute_passes_params_to_store(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)
    tool.execute(query="search term", course_name="MCP", lesson_number=3)
    mock_vector_store.search.assert_called_once_with(
        query="search term", course_name="MCP", lesson_number=3
    )
