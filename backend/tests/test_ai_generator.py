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

    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="rag"
    )
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
def test_round_0_followup_includes_tools(MockAnthropic):
    """After round 0, the follow-up call includes tools to allow a potential second tool call."""
    client = MockAnthropic.return_value

    tool_block = _make_tool_use_block("t1", "search_course_content", {"query": "x"})
    first_resp = Mock(content=[tool_block], stop_reason="tool_use")
    second_resp = Mock(content=[_make_text_block("done")], stop_reason="end_turn")
    client.messages.create.side_effect = [first_resp, second_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "res"

    tools = [{"name": "s"}]
    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q", tools=tools, tool_manager=tool_manager)

    second_call_kwargs = client.messages.create.call_args_list[1][1]
    assert second_call_kwargs["tools"] == tools
    assert second_call_kwargs["tool_choice"] == {"type": "auto"}


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


# ── Sequential multi-tool calling ─────────────────────────────────────


@patch("ai_generator.anthropic.Anthropic")
def test_two_sequential_tool_calls(MockAnthropic):
    """Two tool-use rounds: execute_tool called twice, 3 total API calls, returns text from 3rd."""
    client = MockAnthropic.return_value

    tool_block_1 = _make_tool_use_block(
        "t1", "get_course_outline", {"course_name": "MCP"}
    )
    tool_block_2 = _make_tool_use_block(
        "t2", "search_course_content", {"query": "lesson 3"}
    )

    resp1 = Mock(content=[tool_block_1], stop_reason="tool_use")
    resp2 = Mock(content=[tool_block_2], stop_reason="tool_use")
    resp3 = Mock(content=[_make_text_block("Combined answer")], stop_reason="end_turn")
    client.messages.create.side_effect = [resp1, resp2, resp3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["outline data", "content data"]

    gen = AIGenerator(api_key="fake", model="m")
    result = gen.generate_response(
        "q", tools=[{"name": "t"}], tool_manager=tool_manager
    )

    assert result == "Combined answer"
    assert client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2
    tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
    tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 3")


@patch("ai_generator.anthropic.Anthropic")
def test_final_round_followup_excludes_tools(MockAnthropic):
    """The 3rd API call (final round follow-up) has no tools/tool_choice."""
    client = MockAnthropic.return_value

    tool_block_1 = _make_tool_use_block(
        "t1", "get_course_outline", {"course_name": "X"}
    )
    tool_block_2 = _make_tool_use_block("t2", "search_course_content", {"query": "y"})

    resp1 = Mock(content=[tool_block_1], stop_reason="tool_use")
    resp2 = Mock(content=[tool_block_2], stop_reason="tool_use")
    resp3 = Mock(content=[_make_text_block("done")], stop_reason="end_turn")
    client.messages.create.side_effect = [resp1, resp2, resp3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["r1", "r2"]

    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q", tools=[{"name": "t"}], tool_manager=tool_manager)

    third_call_kwargs = client.messages.create.call_args_list[2][1]
    assert "tools" not in third_call_kwargs
    assert "tool_choice" not in third_call_kwargs


@patch("ai_generator.anthropic.Anthropic")
def test_messages_accumulate_across_rounds(MockAnthropic):
    """The 3rd call messages contain: user, assistant(tool1), user(result1), assistant(tool2), user(result2)."""
    client = MockAnthropic.return_value

    tool_block_1 = _make_tool_use_block(
        "t1", "get_course_outline", {"course_name": "X"}
    )
    tool_block_2 = _make_tool_use_block("t2", "search_course_content", {"query": "y"})

    resp1 = Mock(content=[tool_block_1], stop_reason="tool_use")
    resp2 = Mock(content=[tool_block_2], stop_reason="tool_use")
    resp3 = Mock(content=[_make_text_block("done")], stop_reason="end_turn")
    client.messages.create.side_effect = [resp1, resp2, resp3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["r1", "r2"]

    gen = AIGenerator(api_key="fake", model="m")
    gen.generate_response("q", tools=[{"name": "t"}], tool_manager=tool_manager)

    third_call_kwargs = client.messages.create.call_args_list[2][1]
    messages = third_call_kwargs["messages"]
    assert len(messages) == 5
    assert messages[0]["role"] == "user"  # original query
    assert messages[1]["role"] == "assistant"  # tool_use 1
    assert messages[2]["role"] == "user"  # tool_result 1
    assert messages[3]["role"] == "assistant"  # tool_use 2
    assert messages[4]["role"] == "user"  # tool_result 2


@patch("ai_generator.anthropic.Anthropic")
def test_single_tool_call_still_works(MockAnthropic):
    """Single tool call: only 2 API calls, 1 tool execution — same as before."""
    client = MockAnthropic.return_value

    tool_block = _make_tool_use_block("t1", "search_course_content", {"query": "rag"})
    resp1 = Mock(content=[tool_block], stop_reason="tool_use")
    resp2 = Mock(content=[_make_text_block("answer")], stop_reason="end_turn")
    client.messages.create.side_effect = [resp1, resp2]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    gen = AIGenerator(api_key="fake", model="m")
    result = gen.generate_response(
        "q", tools=[{"name": "t"}], tool_manager=tool_manager
    )

    assert result == "answer"
    assert client.messages.create.call_count == 2
    assert tool_manager.execute_tool.call_count == 1


@patch("ai_generator.anthropic.Anthropic")
def test_tool_execution_error_sent_to_claude(MockAnthropic):
    """When execute_tool raises, error string is sent as tool_result and Claude still responds."""
    client = MockAnthropic.return_value

    tool_block = _make_tool_use_block("t1", "search_course_content", {"query": "x"})
    resp1 = Mock(content=[tool_block], stop_reason="tool_use")
    resp2 = Mock(content=[_make_text_block("handled error")], stop_reason="end_turn")
    client.messages.create.side_effect = [resp1, resp2]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("connection failed")

    gen = AIGenerator(api_key="fake", model="m")
    result = gen.generate_response(
        "q", tools=[{"name": "t"}], tool_manager=tool_manager
    )

    assert result == "handled error"
    second_call_kwargs = client.messages.create.call_args_list[1][1]
    tool_result_content = second_call_kwargs["messages"][-1]["content"][0]["content"]
    assert "Tool execution error: connection failed" in tool_result_content


@patch("ai_generator.anthropic.Anthropic")
def test_no_tool_use_first_round_exits_immediately(MockAnthropic):
    """When Claude doesn't use tools, only 1 API call is made."""
    client = MockAnthropic.return_value

    resp = Mock(content=[_make_text_block("direct answer")], stop_reason="end_turn")
    client.messages.create.return_value = resp

    tool_manager = MagicMock()
    gen = AIGenerator(api_key="fake", model="m")
    result = gen.generate_response(
        "q", tools=[{"name": "t"}], tool_manager=tool_manager
    )

    assert result == "direct answer"
    assert client.messages.create.call_count == 1
    tool_manager.execute_tool.assert_not_called()


@patch("ai_generator.anthropic.Anthropic")
def test_max_tool_rounds_limits_iterations(MockAnthropic):
    """Loop is bounded: exactly 3 API calls for 2 tool rounds, not more."""
    client = MockAnthropic.return_value

    tool_block_1 = _make_tool_use_block(
        "t1", "get_course_outline", {"course_name": "X"}
    )
    tool_block_2 = _make_tool_use_block("t2", "search_course_content", {"query": "y"})

    resp1 = Mock(content=[tool_block_1], stop_reason="tool_use")
    resp2 = Mock(content=[tool_block_2], stop_reason="tool_use")
    resp3 = Mock(content=[_make_text_block("final")], stop_reason="end_turn")
    client.messages.create.side_effect = [resp1, resp2, resp3]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["r1", "r2"]

    gen = AIGenerator(api_key="fake", model="m")
    result = gen.generate_response(
        "q", tools=[{"name": "t"}], tool_manager=tool_manager
    )

    assert result == "final"
    assert client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2
