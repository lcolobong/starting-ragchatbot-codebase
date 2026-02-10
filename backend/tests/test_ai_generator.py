"""Tests for AIGenerator tool-calling mechanics."""

from unittest.mock import patch, MagicMock, Mock, call
from ai_generator import AIGenerator


def _make_text_block(text):
    block = Mock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id, name, tool_input):
    block = Mock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = tool_input
    return block


# ── Direct (no-tool) responses ─────────────────────────────────────────

@patch("ai_generator.anthropic.Anthropic")
def test_direct_response_no_tools(MockAnthropic):
    client = MockAnthropic.return_value
    client.messages.create.return_value = Mock(
        content=[_make_text_block("Hello!")],
        stop_reason="end_turn",
    )
    gen = AIGenerator(api_key="fake", model="test-model")
    result = gen.generate_response("hi")
    assert result == "Hello!"


@patch("ai_generator.anthropic.Anthropic")
def test_tools_passed_to_api_with_auto_choice(MockAnthropic):
    client = MockAnthropic.return_value
    client.messages.create.return_value = Mock(
        content=[_make_text_block("ok")],
        stop_reason="end_turn",
    )
    gen = AIGenerator(api_key="fake", model="m")
    tools = [{"name": "search", "input_schema": {}}]
    gen.generate_response("q", tools=tools)

    kwargs = client.messages.create.call_args[1]
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == {"type": "auto"}


@patch("ai_generator.anthropic.Anthropic")
def test_no_tools_means_no_tool_params(MockAnthropic):
    client = MockAnthropic.return_value
    client.messages.create.return_value = Mock(
        content=[_make_text_block("ok")],
        stop_reason="end_turn",
    )
    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q")

    kwargs = client.messages.create.call_args[1]
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


# ── Conversation history in system prompt ───────────────────────────────

@patch("ai_generator.anthropic.Anthropic")
def test_history_appended_to_system_prompt(MockAnthropic):
    client = MockAnthropic.return_value
    client.messages.create.return_value = Mock(
        content=[_make_text_block("ok")],
        stop_reason="end_turn",
    )
    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q", conversation_history="User: hi\nAI: hello")

    kwargs = client.messages.create.call_args[1]
    expected_suffix = "\n\nPrevious conversation:\nUser: hi\nAI: hello"
    assert kwargs["system"].endswith(expected_suffix)
    assert kwargs["system"].startswith(AIGenerator.SYSTEM_PROMPT)


@patch("ai_generator.anthropic.Anthropic")
def test_no_history_uses_plain_system_prompt(MockAnthropic):
    client = MockAnthropic.return_value
    client.messages.create.return_value = Mock(
        content=[_make_text_block("ok")],
        stop_reason="end_turn",
    )
    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q", conversation_history=None)

    kwargs = client.messages.create.call_args[1]
    assert kwargs["system"] == AIGenerator.SYSTEM_PROMPT


@patch("ai_generator.anthropic.Anthropic")
def test_empty_history_uses_plain_system_prompt(MockAnthropic):
    client = MockAnthropic.return_value
    client.messages.create.return_value = Mock(
        content=[_make_text_block("ok")],
        stop_reason="end_turn",
    )
    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q", conversation_history="")

    kwargs = client.messages.create.call_args[1]
    assert kwargs["system"] == AIGenerator.SYSTEM_PROMPT


# ── Tool-use round-trip ─────────────────────────────────────────────────

@patch("ai_generator.anthropic.Anthropic")
def test_tool_use_executes_tool_via_manager(MockAnthropic):
    client = MockAnthropic.return_value

    tool_block = _make_tool_use_block("t1", "search_course_content", {"query": "rag"})
    first_resp = Mock(
        content=[tool_block],
        stop_reason="tool_use",
    )
    second_resp = Mock(
        content=[_make_text_block("Final answer")],
        stop_reason="end_turn",
    )
    client.messages.create.side_effect = [first_resp, second_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    gen = AIGenerator(api_key="fake", model="m")
    tools = [{"name": "search_course_content"}]
    result = gen.generate_response("q", tools=tools, tool_manager=tool_manager)

    tool_manager.execute_tool.assert_called_once_with("search_course_content", query="rag")
    assert result == "Final answer"


@patch("ai_generator.anthropic.Anthropic")
def test_tool_result_format_sent_to_api(MockAnthropic):
    client = MockAnthropic.return_value

    tool_block = _make_tool_use_block("t42", "search_course_content", {"query": "x"})
    first_resp = Mock(content=[tool_block], stop_reason="tool_use")
    second_resp = Mock(content=[_make_text_block("done")], stop_reason="end_turn")
    client.messages.create.side_effect = [first_resp, second_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result_text"

    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q", tools=[{"name": "s"}], tool_manager=tool_manager)

    second_call_kwargs = client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]
    # Last message should be user role with tool_result
    tool_result_msg = messages[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "t42"
    assert tool_result_msg["content"][0]["content"] == "result_text"


@patch("ai_generator.anthropic.Anthropic")
def test_second_call_excludes_tools(MockAnthropic):
    client = MockAnthropic.return_value

    tool_block = _make_tool_use_block("t1", "search_course_content", {"query": "x"})
    first_resp = Mock(content=[tool_block], stop_reason="tool_use")
    second_resp = Mock(content=[_make_text_block("done")], stop_reason="end_turn")
    client.messages.create.side_effect = [first_resp, second_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "res"

    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q", tools=[{"name": "s"}], tool_manager=tool_manager)

    second_call_kwargs = client.messages.create.call_args_list[1][1]
    assert "tools" not in second_call_kwargs
    assert "tool_choice" not in second_call_kwargs


@patch("ai_generator.anthropic.Anthropic")
def test_tool_use_without_manager_returns_text(MockAnthropic):
    """When stop_reason is tool_use but no tool_manager is provided,
    falls through to content[0].text."""
    client = MockAnthropic.return_value

    text_block = _make_text_block("fallback text")
    tool_block = _make_tool_use_block("t1", "search", {"q": "x"})
    resp = Mock(content=[text_block, tool_block], stop_reason="tool_use")
    client.messages.create.return_value = resp

    gen = AIGenerator(api_key="fake", model="m")
    result = gen.generate_response("q", tools=[{"name": "s"}], tool_manager=None)
    assert result == "fallback text"


# ── Base params ─────────────────────────────────────────────────────────

@patch("ai_generator.anthropic.Anthropic")
def test_base_params_always_included(MockAnthropic):
    client = MockAnthropic.return_value
    client.messages.create.return_value = Mock(
        content=[_make_text_block("ok")],
        stop_reason="end_turn",
    )
    gen = AIGenerator(api_key="fake", model="test-model")
    gen.generate_response("q")

    kwargs = client.messages.create.call_args[1]
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 800
