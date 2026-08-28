from app.llm.anthropic_chat import to_anthropic_messages


def test_groups_consecutive_tool_results_into_one_user_message():
    system, converted = to_anthropic_messages(
        [
            {"role": "system", "content": "Be careful."},
            {"role": "user", "content": "When does Harbor launch?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_uploads",
                            "arguments": '{"query":"Harbor"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_uploads",
                "content": "12 March 2027",
            },
        ]
    )
    assert system == "Be careful."
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"][0]["type"] == "tool_use"
    assert converted[1]["content"][0]["input"] == {"query": "Harbor"}
    assert converted[2]["role"] == "user"
    assert converted[2]["content"][0]["type"] == "tool_result"
    assert converted[2]["content"][0]["tool_use_id"] == "call-1"
