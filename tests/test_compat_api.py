"""Offline coverage for the third-party wire-shape contract.

These run without a cluster: each validator is a pure function over an
already-fetched response, so the whole contract is exercised in CI.
"""

from __future__ import annotations

import json
from typing import Any

from skulk_test_harness.compat_api import (
    HttpOutcome,
    parse_sse,
    validate_anthropic_message,
    validate_ollama_chat,
    validate_ollama_chat_stream,
    validate_ollama_generate,
    validate_ollama_tags,
    validate_ollama_version,
    validate_openai_chat_completion,
    validate_openai_chat_stream,
    validate_openai_embeddings,
    validate_openai_error_envelope,
    validate_openai_model_list,
    validate_success_body_present,
)


def outcome(status: int, body: Any | str | None, *, raw: str | None = None) -> HttpOutcome:
    """Build an HttpOutcome from a payload or a raw string."""

    if raw is not None:
        try:
            parsed = json.loads(raw)
            error = None
        except ValueError as exc:
            parsed = None
            error = str(exc)
        return HttpOutcome(status_code=status, text=raw, json_body=parsed, json_error=error)
    if body is None:
        return HttpOutcome(status_code=status, text="", json_body=None)
    text = json.dumps(body)
    return HttpOutcome(status_code=status, text=text, json_body=body)


def chat_completion(**overrides: Any) -> dict[str, Any]:
    """A well-formed OpenAI chat completion."""

    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1787000000,
        "model": "org/model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    payload.update(overrides)
    return payload


class TestSuccessBodyInvariant:
    """A success status must carry a parseable body."""

    def test_empty_body_on_200_is_a_finding(self) -> None:
        # This is the exact shape observed on a degraded cluster: the endpoint
        # answered 200 with zero bytes and every OpenAI client broke on parse.
        problems = validate_success_body_present(outcome(200, None), "POST /x")
        assert problems
        assert "empty body" in problems[0]

    def test_unparseable_body_on_200_is_a_finding(self) -> None:
        problems = validate_success_body_present(
            outcome(200, None, raw="not json at all"), "POST /x"
        )
        assert problems
        assert "not valid JSON" in problems[0]

    def test_empty_body_on_a_failure_status_is_not_this_check(self) -> None:
        assert validate_success_body_present(outcome(503, None), "POST /x") == []

    def test_chat_completion_surfaces_the_empty_body(self) -> None:
        problems = validate_openai_chat_completion(outcome(200, None))
        assert any("empty body" in problem for problem in problems)


class TestOpenAiErrorEnvelope:
    def test_accepts_the_documented_envelope(self) -> None:
        body = {"error": {"message": "No instance found", "type": "Not Found", "code": 404}}
        assert validate_openai_error_envelope(outcome(404, body), "POST /x", expected_status=404) == []

    def test_rejects_a_success_status(self) -> None:
        problems = validate_openai_error_envelope(outcome(200, chat_completion()), "POST /x")
        assert any("expected a failure status" in problem for problem in problems)

    def test_rejects_a_bare_status_with_no_envelope(self) -> None:
        problems = validate_openai_error_envelope(outcome(500, {"detail": "boom"}), "POST /x")
        assert any("'error' object" in problem for problem in problems)

    def test_reports_an_unexpected_status(self) -> None:
        body = {"error": {"message": "nope", "type": "Bad Request"}}
        problems = validate_openai_error_envelope(
            outcome(400, body), "POST /x", expected_status=404
        )
        assert any("expected HTTP 404" in problem for problem in problems)


class TestOpenAiModelList:
    def test_accepts_a_well_formed_listing(self) -> None:
        body = {
            "object": "list",
            "data": [
                {
                    "id": "org/model",
                    "object": "model",
                    "created": 1787000000,
                    "owned_by": "skulk",
                }
            ],
        }
        assert validate_openai_model_list(outcome(200, body)) == []

    def test_requires_serving_models_to_appear(self) -> None:
        body = {"object": "list", "data": []}
        problems = validate_openai_model_list(
            outcome(200, body), expected_model_ids=("org/serving",)
        )
        assert any("omits 'org/serving'" in problem for problem in problems)

    def test_rejects_a_wrong_object_discriminator(self) -> None:
        body = {"object": "models", "data": []}
        problems = validate_openai_model_list(outcome(200, body))
        assert any("must be 'list'" in problem for problem in problems)

    def test_rejects_entries_missing_required_fields(self) -> None:
        body = {"object": "list", "data": [{"id": "org/model"}]}
        problems = validate_openai_model_list(outcome(200, body))
        joined = " ".join(problems)
        assert "object" in joined and "created" in joined and "owned_by" in joined


class TestOpenAiChatCompletion:
    def test_accepts_a_well_formed_completion(self) -> None:
        assert validate_openai_chat_completion(
            outcome(200, chat_completion()), expected_model_id="org/model"
        ) == []

    def test_requires_the_model_to_be_echoed(self) -> None:
        problems = validate_openai_chat_completion(
            outcome(200, chat_completion()), expected_model_id="org/other"
        )
        assert any("must echo the requested model" in problem for problem in problems)

    def test_rejects_an_empty_assistant_message(self) -> None:
        payload = chat_completion()
        payload["choices"][0]["message"]["content"] = "   "
        problems = validate_openai_chat_completion(outcome(200, payload))
        assert any("non-empty content or tool_calls" in problem for problem in problems)

    def test_accepts_a_tool_call_with_no_text(self) -> None:
        payload = chat_completion()
        payload["choices"][0]["message"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        }
        payload["choices"][0]["finish_reason"] = "tool_calls"
        assert validate_openai_chat_completion(outcome(200, payload)) == []

    def test_rejects_an_unknown_finish_reason(self) -> None:
        payload = chat_completion()
        payload["choices"][0]["finish_reason"] = "finished"
        problems = validate_openai_chat_completion(outcome(200, payload))
        assert any("finish_reason" in problem for problem in problems)

    def test_rejects_inconsistent_usage_totals(self) -> None:
        payload = chat_completion()
        payload["usage"] = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 99}
        problems = validate_openai_chat_completion(outcome(200, payload))
        assert any("total_tokens must equal" in problem for problem in problems)


class TestSseParsing:
    def test_extracts_events_and_the_done_sentinel(self) -> None:
        raw = 'data: {"a":1}\n\ndata: {"a":2}\n\ndata: [DONE]\n'
        stream = parse_sse(raw)
        assert stream.events == ('{"a":1}', '{"a":2}')
        assert stream.saw_done_sentinel is True
        assert stream.malformed_lines == ()

    def test_ignores_comment_lines(self) -> None:
        stream = parse_sse(': keep-alive\ndata: {"a":1}\ndata: [DONE]\n')
        assert stream.events == ('{"a":1}',)

    def test_records_lines_that_are_not_sse_framed(self) -> None:
        stream = parse_sse('{"a":1}\ndata: [DONE]\n')
        assert stream.malformed_lines == ('{"a":1}',)


class TestOpenAiChatStream:
    @staticmethod
    def chunk(content: str, finish: str | None = None) -> str:
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1787000000,
            "model": "org/model",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
        }
        return "data: " + json.dumps(payload)

    def test_accepts_a_well_formed_stream(self) -> None:
        raw = "\n\n".join([self.chunk("he"), self.chunk("llo", "stop"), "data: [DONE]"])
        problems, text = validate_openai_chat_stream(outcome(200, None, raw=raw))
        assert problems == []
        assert text == "hello"

    def test_requires_the_done_sentinel(self) -> None:
        raw = "\n\n".join([self.chunk("he"), self.chunk("llo", "stop")])
        problems, _ = validate_openai_chat_stream(outcome(200, None, raw=raw))
        assert any("[DONE]" in problem for problem in problems)

    def test_rejects_ndjson_framing(self) -> None:
        raw = '{"object":"chat.completion.chunk"}\ndata: [DONE]'
        problems, _ = validate_openai_chat_stream(outcome(200, None, raw=raw))
        assert any("not SSE data" in problem for problem in problems)

    def test_rejects_a_wrong_chunk_discriminator(self) -> None:
        payload = {"object": "chat.completion", "choices": [{"delta": {"content": "x"}}]}
        raw = "data: " + json.dumps(payload) + "\n\ndata: [DONE]"
        problems, _ = validate_openai_chat_stream(outcome(200, None, raw=raw), min_chunks=1)
        assert any("chat.completion.chunk" in problem for problem in problems)

    def test_reports_a_stream_that_carried_no_text(self) -> None:
        raw = "\n\n".join([self.chunk(""), self.chunk("", "stop"), "data: [DONE]"])
        problems, text = validate_openai_chat_stream(outcome(200, None, raw=raw))
        assert text == ""
        assert any("assembled no content" in problem for problem in problems)


class TestOpenAiEmbeddings:
    def test_accepts_a_well_formed_response(self) -> None:
        body = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "org/embed",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
        assert validate_openai_embeddings(outcome(200, body)) == []

    def test_rejects_a_non_numeric_vector(self) -> None:
        body = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": ["a"]}],
            "model": "org/embed",
        }
        problems = validate_openai_embeddings(outcome(200, body))
        assert any("only numbers" in problem for problem in problems)


class TestAnthropicMessage:
    @staticmethod
    def message(**overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "org/model",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        payload.update(overrides)
        return payload

    def test_accepts_a_well_formed_message(self) -> None:
        assert validate_anthropic_message(
            outcome(200, self.message()), expected_model_id="org/model"
        ) == []

    def test_rejects_openai_shape_on_the_anthropic_route(self) -> None:
        # A client reads content[].text, so a chat-completions body here is
        # useless even though it is valid JSON carrying the same answer.
        problems = validate_anthropic_message(outcome(200, chat_completion()))
        joined = " ".join(problems)
        assert "type" in joined and "content" in joined

    def test_rejects_empty_text_blocks(self) -> None:
        problems = validate_anthropic_message(
            outcome(200, self.message(content=[{"type": "text", "text": "  "}]))
        )
        assert any("all empty" in problem for problem in problems)

    def test_rejects_an_unknown_stop_reason(self) -> None:
        problems = validate_anthropic_message(outcome(200, self.message(stop_reason="done")))
        assert any("stop_reason" in problem for problem in problems)

    def test_allows_a_null_stop_reason(self) -> None:
        assert validate_anthropic_message(outcome(200, self.message(stop_reason=None))) == []


class TestOllamaSurface:
    def test_version_requires_a_version_string(self) -> None:
        assert validate_ollama_version(outcome(200, {"version": "0.1.0"}), "/ollama/api/version") == []
        problems = validate_ollama_version(outcome(200, {}), "/ollama/api/version")
        assert any("version" in problem for problem in problems)

    def test_tags_accepts_a_well_formed_listing(self) -> None:
        body = {
            "models": [
                {
                    "name": "org/model",
                    "model": "org/model",
                    "modified_at": "2026-08-21T00:00:00Z",
                    "size": 123,
                    "digest": "sha256:0",
                    "details": {"family": "qwen"},
                }
            ]
        }
        assert validate_ollama_tags(outcome(200, body), "/ollama/api/tags") == []

    def test_tags_requires_serving_models_to_appear(self) -> None:
        problems = validate_ollama_tags(
            outcome(200, {"models": []}),
            "/ollama/api/tags",
            expected_model_ids=("org/serving",),
        )
        assert any("omits 'org/serving'" in problem for problem in problems)

    def test_chat_accepts_a_well_formed_response(self) -> None:
        body = {
            "model": "org/model",
            "created_at": "2026-08-21T00:00:00Z",
            "message": {"role": "assistant", "content": "hi"},
            "done": True,
        }
        assert validate_ollama_chat(outcome(200, body), "/ollama/api/chat") == []

    def test_chat_requires_done(self) -> None:
        body = {
            "model": "org/model",
            "created_at": "2026-08-21T00:00:00Z",
            "message": {"role": "assistant", "content": "hi"},
        }
        problems = validate_ollama_chat(outcome(200, body), "/ollama/api/chat")
        assert any("done" in problem for problem in problems)

    def test_generate_requires_a_response_field(self) -> None:
        body = {"model": "org/model", "response": "", "done": True}
        problems = validate_ollama_generate(outcome(200, body), "/ollama/api/generate")
        assert any("response" in problem for problem in problems)

    def test_chat_stream_accepts_ndjson(self) -> None:
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "he"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": "llo"}, "done": True}),
        ]
        problems, text = validate_ollama_chat_stream(
            outcome(200, None, raw="\n".join(lines)), "/ollama/api/chat"
        )
        assert problems == []
        assert text == "hello"

    def test_chat_stream_rejects_sse_framing(self) -> None:
        raw = 'data: {"message":{"content":"x"},"done":true}'
        problems, _ = validate_ollama_chat_stream(
            outcome(200, None, raw=raw), "/ollama/api/chat"
        )
        assert any("SSE framing" in problem for problem in problems)

    def test_chat_stream_requires_a_terminal_done(self) -> None:
        raw = json.dumps({"message": {"content": "hi"}, "done": False})
        problems, _ = validate_ollama_chat_stream(
            outcome(200, None, raw=raw), "/ollama/api/chat"
        )
        assert any("done=true" in problem for problem in problems)
