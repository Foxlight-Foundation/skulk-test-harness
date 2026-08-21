"""Drive a real third-party application against a Skulk cluster.

The compatibility probes assert the wire contract. This module asserts the
thing an operator actually cares about: that a real application, installed the
way its own documentation says to install it, works when pointed at the
cluster. The two are not the same test. An application exercises the API
through its own client library, its own retry policy and its own assumptions,
and it fails in ways a hand-written probe does not reach.

AnythingLLM is the first such application because it exercises both halves of
the OpenAI surface in one journey: chat completions for the conversation, and
embeddings for document indexing. A cluster that answers chat but cannot embed
looks healthy right up until a user uploads a file.

Container lifecycle uses `subprocess` with argv built by pure functions, in the
same shape as the SSH helpers in `chaos`, rather than adding a Docker SDK
dependency for one suite. The argv builders and the response validators are
pure and unit-tested; only the process and HTTP calls are not.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_IMAGE = "mintplexlabs/anythingllm"
"""Upstream image. Pinning a digest would be better for reproducibility but
would also silently stop testing what users actually install."""

DEFAULT_CONTAINER_NAME = "skulk-harness-anythingllm"
"""Fixed name so a leaked container from an interrupted run is found and
removed by the next one rather than accumulating."""

DEFAULT_HOST_PORT = 3101
"""Deliberately not 3001, so a run never collides with an operator's own
AnythingLLM on the default port."""

GROUNDING_FACT_TITLE = "skulk-harness-grounding-note.txt"
"""Title given to the document uploaded for the retrieval check."""


def grounding_document(token: str) -> str:
    """Build a document whose content cannot be answered from model priors.

    The token is generated per run, so a correct answer proves retrieval
    happened rather than the model recognising something from training.
    """

    return (
        "Foxlight internal note. The Skulk integrations qualification cluster "
        f"is codenamed {token}. Its designated maintenance window is the third "
        "Thursday of each month at 0300 UTC. Fleet batteries must not be "
        "scheduled during that window."
    )


def docker_run_argv(
    *,
    image: str,
    container_name: str,
    host_port: int,
    storage_dir: str,
    api_base_url: str,
    chat_model_id: str,
    embedding_model_id: str | None,
    context_window: int,
    api_key: str = "skulk",
) -> list[str]:
    """Build the argv that starts AnythingLLM pointed at the cluster.

    This mirrors the Docker recipe the dashboard's Integrations page generates,
    so the suite tests the configuration operators are actually handed. The
    storage mount is mandatory: without it the container exits during startup
    while resolving its own paths, which is a confusing failure to debug.
    """

    argv = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{host_port}:3001",
        "-v",
        f"{storage_dir}:/app/server/storage",
        "-e",
        "STORAGE_DIR=/app/server/storage",
        "-e",
        "LLM_PROVIDER=generic-openai",
        "-e",
        f"GENERIC_OPEN_AI_BASE_PATH={api_base_url.rstrip('/')}/v1",
        "-e",
        f"GENERIC_OPEN_AI_MODEL_PREF={chat_model_id}",
        "-e",
        f"GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT={context_window}",
        "-e",
        f"GENERIC_OPEN_AI_API_KEY={api_key}",
        "-e",
        "VECTOR_DB=lancedb",
        "-e",
        "DISABLE_TELEMETRY=true",
    ]
    if embedding_model_id:
        argv.extend(
            [
                "-e",
                "EMBEDDING_ENGINE=generic-openai",
                "-e",
                f"EMBEDDING_BASE_PATH={api_base_url.rstrip('/')}/v1",
                "-e",
                f"EMBEDDING_MODEL_PREF={embedding_model_id}",
                "-e",
                f"GENERIC_OPEN_AI_EMBEDDING_API_KEY={api_key}",
            ]
        )
    argv.append(image)
    return argv


def docker_remove_argv(container_name: str) -> list[str]:
    """Build the argv that force-removes the container, running or not."""

    return ["docker", "rm", "-f", container_name]


def docker_logs_argv(container_name: str, *, tail: int = 80) -> list[str]:
    """Build the argv that captures container logs for failure evidence."""

    return ["docker", "logs", "--tail", str(tail), container_name]


def validate_chat_payload(payload: Any) -> list[str]:
    """Validate AnythingLLM's chat response.

    Checks the application's own contract rather than Skulk's, because that is
    what tells us the integration worked: AnythingLLM reports its own error
    text in `error` and still returns HTTP 200, so a bare status check would
    call a broken integration healthy.
    """

    problems: list[str] = []
    if not isinstance(payload, dict):
        return [f"chat response must be a JSON object, got {type(payload).__name__}"]
    if payload.get("error"):
        problems.append(f"AnythingLLM reported an error: {payload['error']!r}")
    text = payload.get("textResponse")
    if not isinstance(text, str) or not text.strip():
        problems.append("chat response carried no textResponse")
    if payload.get("type") not in {"textResponse", "abort"}:
        problems.append(f"unexpected response type {payload.get('type')!r}")
    return problems


def validate_grounded_payload(
    payload: Any,
    *,
    required_token: str,
    expect_sources: bool = True,
) -> list[str]:
    """Validate a retrieval-grounded answer.

    A correct answer here proves the whole path: the document was embedded
    through the cluster's embedding model, stored, retrieved, and fed back into
    a chat completion served by the cluster.
    """

    problems = validate_chat_payload(payload)
    if not isinstance(payload, dict):
        return problems
    text = payload.get("textResponse")
    if isinstance(text, str) and required_token.lower() not in text.lower():
        problems.append(
            f"grounded answer omits {required_token!r}, so retrieval did not reach "
            f"the model: {text[:200]!r}"
        )
    if expect_sources:
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            problems.append(
                "grounded answer cited no sources, so the answer was not retrieval "
                "backed even if the text looks right"
            )
    return problems


def validate_provider_settings(payload: Any, *, api_base_url: str) -> list[str]:
    """Assert the application really is pointed at this cluster.

    Guards against the most embarrassing false pass: the application answering
    from some other provider that happened to be configured, so the suite
    reports success while never touching Skulk.
    """

    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["settings response must be a JSON object"]
    results = payload.get("results")
    settings = results if isinstance(results, dict) else payload
    expected = api_base_url.rstrip("/") + "/v1"
    base = settings.get("EmbeddingBasePath")
    if isinstance(base, str) and base and base.rstrip("/") != expected:
        problems.append(
            f"embedding base path is {base!r}, expected {expected!r}; the app is "
            "not embedding through this cluster"
        )
    return problems


@dataclass
class ClientAppJourney:
    """Outcome of driving one application end to end."""

    problems: list[str] = field(default_factory=list)
    steps_run: int = 0
    chat_text: str = ""
    grounded_text: str = ""
    vector_count: int = 0
    container_logs: str = ""
    elapsed_s: float = 0.0

    def record(self, problems: list[str]) -> None:
        """Fold one step's problems into the journey."""

        self.steps_run += 1
        self.problems.extend(problems)


def run_command(argv: list[str], *, timeout_s: float = 300.0) -> tuple[int, str]:
    """Run a command and return its exit code and combined output."""

    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        return (127, f"{argv[0]} not found on PATH")
    except subprocess.TimeoutExpired:
        return (124, f"timed out after {timeout_s}s: {' '.join(argv)}")
    return (completed.returncode, (completed.stdout or "") + (completed.stderr or ""))


def wait_until_healthy(
    base_url: str, *, timeout_s: float, poll_interval_s: float = 3.0
) -> bool:
    """Poll the application's ping endpoint until it answers or time runs out."""

    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=10.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{base_url}/api/ping")
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(poll_interval_s)
    return False


def drive_anythingllm(
    base_url: str,
    *,
    api_base_url: str,
    grounding_token: str,
    workspace_slug: str = "skulk-compat",
    request_timeout_s: float = 300.0,
    embed_document: bool = True,
) -> ClientAppJourney:
    """Drive AnythingLLM through its documented developer API.

    Uses the application's own API rather than browser automation because that
    is the surface its behaviour is specified on, and because a browser journey
    would test AnythingLLM's UI rather than the integration.
    """

    journey = ClientAppJourney()
    started = time.perf_counter()
    with httpx.Client(base_url=base_url, timeout=request_timeout_s) as client:
        try:
            settings = client.get("/api/setup-complete")
            journey.record(
                validate_provider_settings(
                    settings.json() if settings.text.strip() else None,
                    api_base_url=api_base_url,
                )
            )
        except (httpx.HTTPError, ValueError) as exc:
            journey.record([f"could not read application settings: {exc}"])

        key_response = client.post("/api/system/generate-api-key")
        try:
            api_key = (key_response.json().get("apiKey") or {}).get("secret")
        except ValueError:
            api_key = None
        if not isinstance(api_key, str) or not api_key:
            journey.record(["could not obtain an application API key"])
            journey.elapsed_s = time.perf_counter() - started
            return journey
        journey.steps_run += 1
        auth = {"Authorization": f"Bearer {api_key}"}

        created = client.post(
            "/api/v1/workspace/new", headers=auth, json={"name": workspace_slug}
        )
        if created.status_code >= 400:
            journey.record(
                [f"workspace creation failed with HTTP {created.status_code}"]
            )
            journey.elapsed_s = time.perf_counter() - started
            return journey
        journey.steps_run += 1

        chat = client.post(
            f"/api/v1/workspace/{workspace_slug}/chat",
            headers=auth,
            json={"message": "Reply with exactly the word: ready", "mode": "chat"},
        )
        chat_payload: Any = None
        try:
            chat_payload = chat.json()
        except ValueError:
            journey.record([f"chat response was not JSON (HTTP {chat.status_code})"])
        if chat_payload is not None:
            journey.record(validate_chat_payload(chat_payload))
            if isinstance(chat_payload, dict):
                journey.chat_text = str(chat_payload.get("textResponse") or "")

        if not embed_document:
            journey.elapsed_s = time.perf_counter() - started
            return journey

        upload = client.post(
            "/api/v1/document/raw-text",
            headers=auth,
            json={
                "textContent": grounding_document(grounding_token),
                "metadata": {"title": GROUNDING_FACT_TITLE},
            },
        )
        document_path: str | None = None
        try:
            documents = (upload.json() or {}).get("documents") or []
            if documents:
                location = documents[0].get("location")
                chunk_source = documents[0].get("chunkSource")
                document_path = location or chunk_source
        except ValueError:
            pass
        if not document_path:
            journey.record(["document upload did not return a usable location"])
            journey.elapsed_s = time.perf_counter() - started
            return journey
        journey.steps_run += 1

        embedded = client.post(
            f"/api/v1/workspace/{workspace_slug}/update-embeddings",
            headers=auth,
            json={"adds": [document_path]},
        )
        if embedded.status_code >= 400:
            journey.record(
                [
                    f"embedding the document failed with HTTP {embedded.status_code}; "
                    "the cluster's embedding surface is the likely cause"
                ]
            )
        else:
            journey.steps_run += 1

        try:
            vectors = client.get("/api/system/system-vectors").json()
            count = vectors.get("vectorCount") if isinstance(vectors, dict) else None
            journey.vector_count = int(count) if isinstance(count, int) else 0
        except (httpx.HTTPError, ValueError, TypeError):
            journey.vector_count = 0
        if journey.vector_count < 1:
            journey.record(
                [
                    "no vectors were stored after embedding, so the document never "
                    "reached the embedding model"
                ]
            )

        grounded = client.post(
            f"/api/v1/workspace/{workspace_slug}/chat",
            headers=auth,
            json={
                "message": (
                    "What is the codename of the integrations qualification "
                    "cluster? Answer with the codename."
                ),
                "mode": "query",
            },
        )
        try:
            grounded_payload = grounded.json()
        except ValueError:
            journey.record(
                [f"grounded response was not JSON (HTTP {grounded.status_code})"]
            )
        else:
            journey.record(
                validate_grounded_payload(
                    grounded_payload, required_token=grounding_token
                )
            )
            if isinstance(grounded_payload, dict):
                journey.grounded_text = str(grounded_payload.get("textResponse") or "")

    journey.elapsed_s = time.perf_counter() - started
    return journey


def summarize_journey(journey: ClientAppJourney, *, limit: int = 10) -> str:
    """Render a bounded human summary of a client-app journey."""

    if not journey.problems:
        return (
            f"{journey.steps_run} steps passed, {journey.vector_count} vector(s) stored"
        )
    shown = journey.problems[:limit]
    remainder = len(journey.problems) - len(shown)
    text = json.dumps(shown)
    if remainder > 0:
        text += f" (+{remainder} more)"
    return text
