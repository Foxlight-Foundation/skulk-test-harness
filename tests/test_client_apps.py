"""Offline coverage for the client-application journey.

The container lifecycle and HTTP calls need a host and a cluster, so they are
only reachable from `run --execute`. Everything else here is pure: the argv the
suite would run, and the verdicts it would reach given an application's
responses. Those are the parts where a mistake would make the suite pass when
the integration is broken, which is the failure mode worth guarding in CI.
"""

from __future__ import annotations

from typing import Any

from skulk_test_harness.client_apps import (
    DEFAULT_CONTAINER_NAME,
    docker_logs_argv,
    docker_remove_argv,
    docker_run_argv,
    grounding_document,
    validate_chat_payload,
    validate_grounded_payload,
    validate_provider_settings,
)


def run_argv(**overrides: Any) -> list[str]:
    """Build a representative docker run argv."""

    kwargs: dict[str, Any] = {
        "image": "mintplexlabs/anythingllm",
        "container_name": DEFAULT_CONTAINER_NAME,
        "host_port": 3101,
        "storage_dir": "/tmp/storage",
        "api_base_url": "http://192.168.1.50:52415",
        "chat_model_id": "org/chat",
        "embedding_model_id": "org/embed",
        "context_window": 8192,
    }
    kwargs.update(overrides)
    return docker_run_argv(**kwargs)


class TestDockerArgv:
    def test_points_the_app_at_the_cluster_openai_surface(self) -> None:
        argv = run_argv()
        joined = " ".join(argv)
        assert "GENERIC_OPEN_AI_BASE_PATH=http://192.168.1.50:52415/v1" in joined
        assert "LLM_PROVIDER=generic-openai" in joined
        assert "GENERIC_OPEN_AI_MODEL_PREF=org/chat" in joined

    def test_mounts_storage_because_the_app_exits_without_it(self) -> None:
        argv = run_argv()
        assert "/tmp/storage:/app/server/storage" in argv
        assert "STORAGE_DIR=/app/server/storage" in " ".join(argv)

    def test_configures_the_embedder_only_when_a_model_is_given(self) -> None:
        with_embed = " ".join(run_argv())
        assert "EMBEDDING_ENGINE=generic-openai" in with_embed
        assert "EMBEDDING_MODEL_PREF=org/embed" in with_embed

        without_embed = " ".join(run_argv(embedding_model_id=None))
        assert "EMBEDDING_ENGINE" not in without_embed
        assert "EMBEDDING_MODEL_PREF" not in without_embed

    def test_does_not_publish_on_the_apps_default_port(self) -> None:
        # Colliding with an operator's own AnythingLLM would be a confusing
        # failure, and worse, could drive their instance instead of ours.
        argv = run_argv()
        assert "3101:3001" in argv

    def test_trailing_slash_in_the_cluster_url_does_not_double_up(self) -> None:
        joined = " ".join(run_argv(api_base_url="http://192.168.1.50:52415/"))
        assert "52415/v1" in joined
        assert "52415//v1" not in joined

    def test_image_is_the_final_argument(self) -> None:
        argv = run_argv()
        assert argv[-1] == "mintplexlabs/anythingllm"

    def test_remove_is_forced_so_a_stopped_container_is_also_cleared(self) -> None:
        assert docker_remove_argv("x") == ["docker", "rm", "-f", "x"]

    def test_logs_argv_bounds_output(self) -> None:
        assert docker_logs_argv("x", tail=20) == ["docker", "logs", "--tail", "20", "x"]


class TestChatValidation:
    def test_accepts_a_normal_answer(self) -> None:
        payload = {"type": "textResponse", "textResponse": "ready", "error": None}
        assert validate_chat_payload(payload) == []

    def test_rejects_an_application_reported_error_despite_http_200(self) -> None:
        # AnythingLLM returns 200 with an error string when the provider call
        # fails, so a status-only check would call a broken integration healthy.
        payload = {
            "type": "textResponse",
            "textResponse": "",
            "error": "Could not respond to message.",
        }
        problems = validate_chat_payload(payload)
        assert any("reported an error" in problem for problem in problems)

    def test_rejects_an_empty_answer(self) -> None:
        payload = {"type": "textResponse", "textResponse": "   ", "error": None}
        assert any("no textResponse" in problem for problem in validate_chat_payload(payload))


class TestGroundedValidation:
    @staticmethod
    def payload(text: str, sources: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "type": "textResponse",
            "textResponse": text,
            "error": None,
            "sources": sources if sources is not None else [{"title": "note.txt"}],
        }

    def test_accepts_an_answer_carrying_the_token_and_a_source(self) -> None:
        problems = validate_grounded_payload(
            self.payload("The codename is VERMILLION-ABC12345."),
            required_token="VERMILLION-ABC12345",
        )
        assert problems == []

    def test_is_case_insensitive_about_the_token(self) -> None:
        problems = validate_grounded_payload(
            self.payload("the codename is vermillion-abc12345"),
            required_token="VERMILLION-ABC12345",
        )
        assert problems == []

    def test_rejects_a_fluent_answer_that_missed_the_document(self) -> None:
        # The whole point of a per-run token: a plausible answer that does not
        # contain it proves retrieval never happened.
        problems = validate_grounded_payload(
            self.payload("I do not have information about that cluster."),
            required_token="VERMILLION-ABC12345",
        )
        assert any("omits" in problem for problem in problems)

    def test_rejects_an_answer_with_no_sources(self) -> None:
        problems = validate_grounded_payload(
            self.payload("The codename is VERMILLION-ABC12345.", sources=[]),
            required_token="VERMILLION-ABC12345",
        )
        assert any("cited no sources" in problem for problem in problems)


class TestProviderSettings:
    def test_accepts_settings_pointed_at_this_cluster(self) -> None:
        payload = {
            "results": {"EmbeddingBasePath": "http://192.168.1.50:52415/v1"}
        }
        assert validate_provider_settings(payload, api_base_url="http://192.168.1.50:52415") == []

    def test_rejects_an_app_pointed_somewhere_else(self) -> None:
        # Guards the most embarrassing false pass: the app answering from a
        # different provider while the suite reports success.
        payload = {"results": {"EmbeddingBasePath": "https://api.openai.com/v1"}}
        problems = validate_provider_settings(
            payload, api_base_url="http://192.168.1.50:52415"
        )
        assert any("not embedding through this cluster" in problem for problem in problems)


class TestGroundingDocument:
    def test_carries_the_token_and_is_not_answerable_from_priors(self) -> None:
        text = grounding_document("VERMILLION-ABC12345")
        assert "VERMILLION-ABC12345" in text
        assert "third Thursday" in text
