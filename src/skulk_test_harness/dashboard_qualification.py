"""Live Playwright qualification of the dashboard's real user journey."""

from __future__ import annotations

import base64
import io
import re
import sys
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from playwright.sync_api import (
    Browser,
    BrowserContext,
    BrowserType,
    Locator,
    Page,
    Request,
    Route,
    sync_playwright,
)

from skulk_test_harness.client import SkulkClient
from skulk_test_harness.echo_phrase import echo_matched, echo_phrase, echo_prompt
from skulk_test_harness.models import (
    DashboardAudioEvidence,
    DashboardExperienceEvidence,
    DashboardJourneyOutcome,
    VisionFixtureEvidence,
)
from skulk_test_harness.vision_fixture import VisionFixture, data_url_sha256

_CAPTURE_AUDIO_FETCH_SCRIPT = """
(() => {
  const nativeFetch = window.fetch.bind(window);
  window.__skulkQualificationAudio = [];
  window.fetch = async (...args) => {
    const response = await nativeFetch(...args);
    const input = args[0];
    const rawUrl = typeof input === "string" ? input : input.url;
    const url = new URL(rawUrl, window.location.href);
    if (url.pathname === "/v1/audio/speech") {
      const copy = response.clone();
      void copy.arrayBuffer().then((buffer) => {
        const bytes = new Uint8Array(buffer);
        let binary = "";
        const chunkSize = 0x8000;
        for (let offset = 0; offset < bytes.length; offset += chunkSize) {
          binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
        }
        window.__skulkQualificationAudio.push({
          size: bytes.length,
          bodyBase64: btoa(binary),
        });
      }).catch((error) => {
        window.__skulkQualificationAudio.push({ error: String(error) });
      });
    }
    return response;
  };
})();
"""


@dataclass
class _JourneyProgress:
    """Mutable record of how far one browser journey advanced.

    The reported outcome is immutable, but it has to be produced on both the
    success and the failure path. Accumulating progress here is what lets a
    failure say which step broke instead of reporting an untouched outcome in
    which every step looks like it never ran.
    """

    model_id: str
    found: bool = False
    download_started: bool = False
    launched: bool = False
    selected: bool = False
    text_chat_passed: bool = False
    vision: VisionFixtureEvidence | None = None
    false_vision_path_offered: bool | None = None
    first_run_consent_prompted: bool = False
    conversation_persisted_after_reload: bool = False
    attachment_persisted_after_reload: bool | None = None
    text_chat_response: str | None = None
    vision_response: str | None = None

    def failure_message(self) -> str | None:
        """Explain a failure the booleans alone cannot, or return None.

        A journey that reaches its assertions and fails them raises nothing,
        so without this the report carries a bare ``passed: false``. The
        response text is what distinguishes a model declining the prompt, or
        answering the wrong question, from a genuinely broken path.
        """

        parts: list[str] = []
        if self.text_chat_response is not None:
            parts.append(
                "chat response did not contain the requested phrase; "
                f"the model replied: {self.text_chat_response!r}"
            )
        if self.vision_response is not None:
            parts.append(
                "vision response did not match the fixture; "
                f"the model replied: {self.vision_response!r}"
            )
        return "; ".join(parts) if parts else None

    def outcome(
        self,
        *,
        passed: bool,
        message: str | None = None,
    ) -> DashboardJourneyOutcome:
        """Freeze the progress recorded so far into a reportable outcome."""

        return DashboardJourneyOutcome(
            model_id=self.model_id,
            found=self.found,
            download_started=self.download_started,
            launched=self.launched,
            selected=self.selected,
            text_chat_passed=self.text_chat_passed,
            vision=self.vision,
            false_vision_path_offered=self.false_vision_path_offered,
            first_run_consent_prompted=self.first_run_consent_prompted,
            conversation_persisted_after_reload=(
                self.conversation_persisted_after_reload
            ),
            attachment_persisted_after_reload=self.attachment_persisted_after_reload,
            passed=passed,
            message=message,
        )


class DashboardQualifier:
    """Drive find, download, launch, select, and chat through the served UI."""

    def __init__(
        self,
        *,
        api_base_url: str,
        artifact_directory: Path,
        poll_interval_s: float,
        model_ready_timeout_s: float,
        abort_check: Callable[[], None] | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.artifact_directory = artifact_directory
        self.poll_interval_s = poll_interval_s
        self.model_ready_timeout_s = model_ready_timeout_s
        self.abort_check = abort_check

    def qualify(
        self,
        *,
        model_id: str,
        vision_contract: str,
        fixture: VisionFixture | None,
    ) -> DashboardJourneyOutcome:
        """Run one browser journey and retain its trace and final screenshot."""

        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        captured_chat_requests: list[dict[str, object]] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
            page = context.new_page()

            def capture_chat_request(request: Request) -> None:
                if request.method != "POST" or not request.url.endswith(
                    "/v1/chat/completions"
                ):
                    return
                try:
                    payload = request.post_data_json
                except Exception:  # noqa: BLE001 - Playwright parser can reject bodies
                    return
                if isinstance(payload, dict):
                    captured_chat_requests.append(
                        {str(key): value for key, value in payload.items()}
                    )

            page.on("request", capture_chat_request)
            # Progress is accumulated as the journey advances rather than
            # assembled at the end, so a failure reports how far it actually
            # got. Reporting the untouched initial outcome made every failure
            # look like it happened at the first click even when find,
            # download, launch, and chat had all succeeded.
            progress = _JourneyProgress(model_id=model_id)
            try:
                return self._run_journey(
                    page,
                    model_id=model_id,
                    vision_contract=vision_contract,
                    fixture=fixture,
                    captured_chat_requests=captured_chat_requests,
                    progress=progress,
                )
            except Exception as exception:  # noqa: BLE001 - report browser boundary
                return progress.outcome(passed=False, message=str(exception))
            finally:
                safe_name = _safe_model_name(model_id)
                active_exception = sys.exception()
                cleanup_error: Exception | None = None
                try:
                    page.screenshot(
                        path=str(self.artifact_directory / f"{safe_name}.final.png"),
                        full_page=True,
                    )
                except Exception as exception:  # noqa: BLE001 - cleanup boundary
                    cleanup_error = exception
                try:
                    context.tracing.stop(
                        path=str(self.artifact_directory / f"{safe_name}.trace.zip")
                    )
                except Exception as exception:  # noqa: BLE001 - cleanup boundary
                    if cleanup_error is None:
                        cleanup_error = exception
                try:
                    browser.close()
                except Exception as exception:  # noqa: BLE001 - cleanup boundary
                    if cleanup_error is None:
                        cleanup_error = exception
                # Artifact failures remain fatal during an otherwise normal
                # journey, but cleanup must never replace an operator signal
                # or another active BaseException with a secondary Playwright
                # "page has been closed" error.
                if active_exception is None and cleanup_error is not None:
                    raise cleanup_error

    def qualify_experience(
        self,
        *,
        model_id: str,
        expected_node_count: int,
    ) -> DashboardExperienceEvidence:
        """Exercise release-critical dashboard surfaces outside model provisioning."""

        settings_opened = False
        settings_saved = False
        topology_visible_nodes = 0
        request_failure_visible = False
        request_retry_passed = False
        webkit_loaded = False
        webkit_text_chat_passed = False
        message: str | None = None
        self.artifact_directory.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as playwright:
            chromium = playwright.chromium.launch(headless=True)
            chromium_context = chromium.new_context(
                viewport={"width": 1440, "height": 1000}
            )
            chromium_context.tracing.start(
                screenshots=True, snapshots=True, sources=True
            )
            chromium_page = chromium_context.new_page()
            try:
                chromium_page.goto(
                    f"{self.api_base_url}/cluster", wait_until="networkidle"
                )
                self._dismiss_first_run_consent(chromium_page)
                settings_opened, settings_saved = self._qualify_settings(chromium_page)
                topology_visible_nodes = self._qualify_topology(
                    chromium_page,
                    expected_node_count=expected_node_count,
                )
                (
                    request_failure_visible,
                    request_retry_passed,
                ) = self._qualify_failure_and_retry(
                    chromium_page,
                    model_id=model_id,
                )
            except Exception as exception:  # noqa: BLE001 - browser report boundary
                message = str(exception)
            finally:
                cleanup_error = _capture_and_close_browser(
                    chromium_page,
                    chromium_context,
                    chromium,
                    screenshot_path=(
                        self.artifact_directory / "dashboard-experience.final.png"
                    ),
                    trace_path=(
                        self.artifact_directory / "dashboard-experience.trace.zip"
                    ),
                )
                if message is None and cleanup_error is not None:
                    message = f"Chromium cleanup failed: {cleanup_error}"

            webkit = playwright.webkit.launch(headless=True)
            webkit_context = webkit.new_context(
                viewport={"width": 1440, "height": 1000}
            )
            webkit_context.tracing.start(screenshots=True, snapshots=True, sources=True)
            webkit_page = webkit_context.new_page()
            try:
                webkit_page.goto(f"{self.api_base_url}/chat", wait_until="networkidle")
                webkit_loaded = True
                self._dismiss_first_run_consent(webkit_page)
                self._select_chat_model(webkit_page, model_id=model_id)
                phrase = echo_phrase()
                message_box = webkit_page.get_by_label("Chat message", exact=True)
                message_box.fill(echo_prompt(phrase))
                webkit_page.get_by_role(
                    "button", name="Send message", exact=True
                ).click()
                response = self._wait_for_assistant(
                    webkit_page,
                    expected=phrase,
                )
                webkit_text_chat_passed = echo_matched(phrase, response)
            except Exception as exception:  # noqa: BLE001 - browser report boundary
                if message is None:
                    message = f"WebKit smoke failed: {exception}"
            finally:
                cleanup_error = _capture_and_close_browser(
                    webkit_page,
                    webkit_context,
                    webkit,
                    screenshot_path=(
                        self.artifact_directory / "dashboard-webkit.final.png"
                    ),
                    trace_path=(self.artifact_directory / "dashboard-webkit.trace.zip"),
                )
                if message is None and cleanup_error is not None:
                    message = f"WebKit cleanup failed: {cleanup_error}"

        passed = (
            settings_opened
            and settings_saved
            and topology_visible_nodes == expected_node_count
            and request_failure_visible
            and request_retry_passed
            and webkit_loaded
            and webkit_text_chat_passed
        )
        return DashboardExperienceEvidence(
            model_id=model_id,
            settings_opened=settings_opened,
            settings_saved=settings_saved,
            topology_expected_nodes=expected_node_count,
            topology_visible_nodes=topology_visible_nodes,
            request_failure_visible=request_failure_visible,
            request_retry_passed=request_retry_passed,
            webkit_loaded=webkit_loaded,
            webkit_text_chat_passed=webkit_text_chat_passed,
            passed=passed,
            message=message,
        )

    def _qualify_settings(self, page: Page) -> tuple[bool, bool]:
        """Open the real Settings drawer and round-trip the generated config."""

        page.get_by_role("button", name="Settings", exact=True).click()
        heading = page.get_by_text("Settings", exact=True)
        heading.wait_for(state="visible", timeout=30_000)
        page.get_by_text("Allow HuggingFace fallback", exact=True).wait_for(
            state="visible", timeout=30_000
        )
        with page.expect_response(
            lambda response: (
                response.request.method == "PUT" and response.url.endswith("/config")
            ),
            timeout=30_000,
        ) as response_info:
            page.get_by_role("button", name="Save", exact=True).click()
        response = response_info.value
        if not response.ok:
            raise RuntimeError(
                f"dashboard Settings save failed with HTTP {response.status}"
            )
        heading.wait_for(state="hidden", timeout=30_000)
        self._check_abort()
        return True, True

    def _qualify_topology(self, page: Page, *, expected_node_count: int) -> int:
        """Require the cluster graph to render every expected fresh member."""

        page.goto(f"{self.api_base_url}/cluster", wait_until="networkidle")
        inspect = page.get_by_role(
            "button", name="Inspect live node diagnostics", exact=True
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self._check_abort()
            count = inspect.count()
            if count == expected_node_count:
                return count
            page.wait_for_timeout(250)
        return inspect.count()

    def _qualify_failure_and_retry(
        self,
        page: Page,
        *,
        model_id: str,
    ) -> tuple[bool, bool]:
        """Inject one browser-visible request failure, then prove recovery."""

        page.goto(f"{self.api_base_url}/chat", wait_until="networkidle")
        self._select_chat_model(page, model_id=model_id)
        message_box = self._start_new_conversation(page, model_id=model_id)
        injected = False

        def fail_once(route: Route, request: Request) -> None:
            nonlocal injected
            if (
                not injected
                and request.method == "POST"
                and request.url.endswith("/v1/chat/completions")
            ):
                injected = True
                route.fulfill(
                    status=503,
                    content_type="application/json",
                    body='{"detail":"qualification injected failure"}',
                )
                return
            route.continue_()

        page.route("**/v1/chat/completions", fail_once)
        message_box.fill("This request is expected to fail once.")
        page.get_by_role("button", name="Send message", exact=True).click()
        failed_response = self._wait_for_assistant(
            page,
            expected="qualification injected failure",
        )
        request_failure_visible = (
            "qualification injected failure" in failed_response.casefold()
        )
        page.unroute("**/v1/chat/completions", fail_once)

        message_box = self._start_new_conversation(page, model_id=model_id)
        phrase = echo_phrase()
        message_box.fill(echo_prompt(phrase))
        page.get_by_role("button", name="Send message", exact=True).click()
        retry_response = self._wait_for_assistant(page, expected=phrase)
        return request_failure_visible, echo_matched(phrase, retry_response)

    def _select_chat_model(self, page: Page, *, model_id: str) -> None:
        """Select an exact mounted model in the shipped chat model control."""

        selector = page.get_by_label("Select chat model", exact=True)
        if selector.count() > 0:
            selector.wait_for(state="visible", timeout=30_000)
            selector.select_option(model_id)
            return
        message = page.get_by_label("Chat message", exact=True)
        message.wait_for(state="visible", timeout=30_000)
        label = model_id.rsplit("/", maxsplit=1)[-1]
        placeholder = message.get_attribute("placeholder") or ""
        if label.casefold() not in placeholder.casefold():
            raise RuntimeError(
                "dashboard did not expose the exact requested chat model; "
                f"expected {label!r} in composer placeholder {placeholder!r}"
            )

    def qualify_audio(
        self,
        *,
        chat_model_id: str,
        speech_synthesis_model: str,
        transcription_model: str,
    ) -> DashboardAudioEvidence:
        """Drive real dashboard TTS and fake-device microphone STT end to end."""

        synthesis_found = False
        synthesis_downloaded = False
        synthesis_launched = False
        transcription_found = False
        transcription_downloaded = False
        transcription_launched = False
        synthesis_request_observed = False
        synthesis_media_type: str | None = None
        synthesis_audio_bytes = 0
        synthesis_audio_sha256: str | None = None
        transcription_request_observed = False
        transcript_matched = False
        message: str | None = None
        fixture_phrase = "release audio bravo hotel seven cedar"
        fixture_path = self.artifact_directory / "dashboard-stt-fixture.wav"
        fixture_source = (
            Path(__file__).parent / "fixtures" / "dashboard-stt-release.wav"
        )
        synthesis_duration_s: float | None = None
        synthesis_rms: float | None = None
        transcription_response_text: str | None = None
        transcription_composer_text: str | None = None
        self.artifact_directory.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=["--autoplay-policy=no-user-gesture-required"],
            )
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
            page = context.new_page()
            page.add_init_script(_CAPTURE_AUDIO_FETCH_SCRIPT)
            try:
                (
                    synthesis_found,
                    synthesis_downloaded,
                    synthesis_launched,
                ) = self._provision_dashboard_model(
                    page,
                    model_id=speech_synthesis_model,
                )
                (
                    transcription_found,
                    transcription_downloaded,
                    transcription_launched,
                ) = self._provision_dashboard_model(
                    page,
                    model_id=transcription_model,
                )
                page.goto(f"{self.api_base_url}/chat", wait_until="networkidle")
                self._select_chat_model(page, model_id=chat_model_id)
                speech_selector = page.get_by_label("Select speech model", exact=True)
                speech_selector.wait_for(state="visible", timeout=30_000)
                speech_selector.select_option(speech_synthesis_model)
                draft = page.get_by_label("Chat message", exact=True)
                draft.fill(fixture_phrase)
                speak = page.get_by_role("button", name="Speak draft", exact=True)
                speak.wait_for(state="visible", timeout=30_000)
                deadline = time.monotonic() + 30
                while speak.is_disabled() and time.monotonic() < deadline:
                    page.wait_for_timeout(250)
                if speak.is_disabled():
                    raise RuntimeError(
                        "dashboard Speak draft control did not become enabled"
                    )
                with page.expect_response(
                    lambda response: (
                        response.request.method == "POST"
                        and response.url.endswith("/v1/audio/speech")
                    ),
                    timeout=self.model_ready_timeout_s * 1000,
                ) as response_info:
                    speak.click()
                speech_response = response_info.value
                if not speech_response.ok:
                    raise RuntimeError(
                        "dashboard speech request failed with HTTP "
                        f"{speech_response.status}"
                    )
                synthesis_request_observed = True
                synthesis_media_type = (
                    speech_response.headers.get("content-type", "")
                    .split(";", maxsplit=1)[0]
                    .strip()
                    .lower()
                )
                if not synthesis_media_type.startswith("audio/"):
                    raise RuntimeError(
                        "dashboard speech response was not audio: "
                        f"{synthesis_media_type!r}"
                    )
                deadline = time.monotonic() + 60
                captured_audio: object = None
                while time.monotonic() < deadline:
                    captured_audio = page.evaluate(
                        "() => window.__skulkQualificationAudio?.at(-1) ?? null"
                    )
                    if isinstance(captured_audio, dict):
                        break
                    self._check_abort()
                    page.wait_for_timeout(250)
                if not isinstance(captured_audio, dict):
                    raise RuntimeError(
                        "dashboard did not finish consuming its speech response"
                    )
                raw_size = captured_audio.get("size")
                raw_body = captured_audio.get("bodyBase64")
                if not isinstance(raw_size, int) or not isinstance(raw_body, str):
                    raise RuntimeError(
                        "dashboard speech capture did not contain byte evidence"
                    )
                audio_body = base64.b64decode(raw_body, validate=True)
                if len(audio_body) != raw_size:
                    raise RuntimeError(
                        "dashboard speech capture byte count did not match its body"
                    )
                synthesis_audio_bytes = len(audio_body)
                synthesis_audio_sha256 = sha256(audio_body).hexdigest()
                if synthesis_audio_bytes < 1024:
                    raise RuntimeError(
                        "dashboard speech response was implausibly short: "
                        f"{synthesis_audio_bytes} bytes"
                    )
                (
                    synthesis_duration_s,
                    synthesis_rms,
                ) = _pcm_wav_duration_and_rms(audio_body)
                if synthesis_duration_s < 0.5 or synthesis_rms < 50:
                    raise RuntimeError(
                        "dashboard speech response was silent or implausibly short: "
                        f"duration={synthesis_duration_s:.3f}s rms={synthesis_rms:.1f}"
                    )
                (self.artifact_directory / "dashboard-tts-output.wav").write_bytes(
                    audio_body
                )
                self._check_abort()
            except Exception as exception:  # noqa: BLE001 - browser report boundary
                message = str(exception)
            finally:
                cleanup_error = _capture_and_close_browser(
                    page,
                    context,
                    browser,
                    screenshot_path=(
                        self.artifact_directory / "dashboard-audio.final.png"
                    ),
                    trace_path=(self.artifact_directory / "dashboard-audio.trace.zip"),
                )
                if message is None and cleanup_error is not None:
                    message = f"dashboard audio cleanup failed: {cleanup_error}"

            if message is None:
                try:
                    fixture_path.write_bytes(fixture_source.read_bytes())
                    (
                        transcription_request_observed,
                        transcript_matched,
                        transcription_response_text,
                        transcription_composer_text,
                    ) = self._qualify_fake_microphone_transcription(
                        playwright.chromium,
                        chat_model_id=chat_model_id,
                        transcription_model=transcription_model,
                        fixture_path=fixture_path,
                        expected_transcript=fixture_phrase,
                    )
                except Exception as exception:  # noqa: BLE001 - browser report boundary
                    message = str(exception)

        passed = (
            synthesis_found
            and synthesis_downloaded
            and synthesis_launched
            and transcription_found
            and transcription_downloaded
            and transcription_launched
            and synthesis_request_observed
            and synthesis_audio_bytes >= 1024
            and synthesis_duration_s is not None
            and synthesis_duration_s >= 0.5
            and synthesis_rms is not None
            and synthesis_rms >= 50
            and transcription_request_observed
            and transcript_matched
        )
        return DashboardAudioEvidence(
            speech_synthesis_model=speech_synthesis_model,
            transcription_model=transcription_model,
            synthesis_found=synthesis_found,
            synthesis_downloaded=synthesis_downloaded,
            synthesis_launched=synthesis_launched,
            transcription_found=transcription_found,
            transcription_downloaded=transcription_downloaded,
            transcription_launched=transcription_launched,
            synthesis_request_observed=synthesis_request_observed,
            synthesis_media_type=synthesis_media_type,
            synthesis_audio_bytes=synthesis_audio_bytes,
            synthesis_audio_sha256=synthesis_audio_sha256,
            synthesis_duration_s=synthesis_duration_s,
            synthesis_rms=synthesis_rms,
            transcription_request_observed=transcription_request_observed,
            transcription_response_text=transcription_response_text,
            transcription_composer_text=transcription_composer_text,
            transcript_matched=transcript_matched,
            passed=passed,
            message=message,
        )

    def _provision_dashboard_model(
        self,
        page: Page,
        *,
        model_id: str,
    ) -> tuple[bool, bool, bool]:
        """Find, download, and launch one model exclusively through dashboard UI."""

        page.goto(f"{self.api_base_url}/model-store", wait_until="networkidle")
        self._dismiss_first_run_consent(page)
        with SkulkClient(self.api_base_url) as client:
            registry = client.get_store_registry()
        already_downloaded = registry is not None and _registry_contains(
            registry, model_id
        )
        launch = page.get_by_role("button", name=f"Launch {model_id}", exact=True)
        if not already_downloaded:
            page.get_by_role("button", name="Find Models", exact=True).click()
            search = page.get_by_label("Search models", exact=True)
            search.fill(model_id)
            download = self._wait_for_download_action(page, model_id=model_id)
            download.click()
            page.get_by_role("button", name="Close", exact=True).click()
            self._wait_for_store_model(model_id)
            page.reload(wait_until="networkidle")
            launch = page.get_by_role("button", name=f"Launch {model_id}", exact=True)
        launch.wait_for(state="visible", timeout=30_000)
        launch.click()
        self._wait_for_ready_instance(model_id)
        return True, True, True

    def _qualify_fake_microphone_transcription(
        self,
        chromium: BrowserType,
        *,
        chat_model_id: str,
        transcription_model: str,
        fixture_path: Path,
        expected_transcript: str,
    ) -> tuple[bool, bool, str, str]:
        """Feed a deterministic WAV through Chromium's fake microphone device."""

        browser = chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={fixture_path}",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            permissions=["microphone"],
        )
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        transcription_observed = False
        transcript_matched = False
        try:
            page.goto(f"{self.api_base_url}/chat", wait_until="networkidle")
            self._dismiss_first_run_consent(page)
            self._select_chat_model(page, model_id=chat_model_id)
            transcription_selector = page.get_by_label(
                "Select transcription model", exact=True
            )
            transcription_selector.wait_for(state="visible", timeout=30_000)
            transcription_selector.select_option(transcription_model)
            start = page.get_by_role("button", name="Start recording", exact=True)
            start.wait_for(state="visible", timeout=30_000)
            with page.expect_response(
                lambda response: (
                    response.request.method == "POST"
                    and response.url.endswith("/v1/audio/transcriptions")
                ),
                timeout=180_000,
            ) as response_info:
                start.click()
                stop = page.get_by_role("button", name="Stop recording", exact=True)
                stop.wait_for(state="visible", timeout=30_000)
                page.wait_for_timeout(_fake_microphone_recording_ms(fixture_path))
                stop.click()
            transcription_response = response_info.value
            transcription_observed = True
            if not transcription_response.ok:
                raise RuntimeError(
                    "dashboard transcription request failed with HTTP "
                    f"{transcription_response.status}: "
                    f"{transcription_response.text()[:400]}"
                )
            response_body = transcription_response.json()
            response_text = (
                response_body.get("text") if isinstance(response_body, dict) else None
            )
            response_transcript = (
                response_text if isinstance(response_text, str) else ""
            )
            if not response_transcript.strip():
                raise RuntimeError(
                    "dashboard transcription response contained no transcript"
                )
            message_box = page.get_by_label("Chat message", exact=True)
            deadline = time.monotonic() + 10
            transcript = ""
            while time.monotonic() < deadline:
                transcript = message_box.input_value()
                if transcript.strip():
                    break
                self._check_abort()
                page.wait_for_timeout(250)
            if not transcript.strip():
                raise RuntimeError(
                    "dashboard received a transcript but did not place it in "
                    "the chat composer"
                )
            transcript_matched = _transcript_matches(
                expected_transcript,
                transcript,
            )
            return (
                transcription_observed,
                transcript_matched,
                response_transcript,
                transcript,
            )
        finally:
            active_exception = sys.exception()
            cleanup_error = _capture_and_close_browser(
                page,
                context,
                browser,
                screenshot_path=self.artifact_directory / "dashboard-stt.final.png",
                trace_path=self.artifact_directory / "dashboard-stt.trace.zip",
            )
            if active_exception is None and cleanup_error is not None:
                raise cleanup_error

    def _run_journey(
        self,
        page: Page,
        *,
        model_id: str,
        vision_contract: str,
        fixture: VisionFixture | None,
        captured_chat_requests: list[dict[str, object]],
        progress: _JourneyProgress,
    ) -> DashboardJourneyOutcome:
        page.goto(f"{self.api_base_url}/model-store", wait_until="networkidle")
        self._check_abort()
        progress.first_run_consent_prompted = self._dismiss_first_run_consent(page)
        page.get_by_role("button", name="Find Models", exact=True).click()
        search = page.get_by_label("Search models", exact=True)
        search.fill(model_id)
        download = self._wait_for_download_action(page, model_id=model_id)
        progress.found = True
        download.click()
        progress.download_started = True
        page.get_by_role("button", name="Close", exact=True).click()
        self._wait_for_store_model(model_id)

        page.reload(wait_until="networkidle")
        launch = page.get_by_role("button", name=f"Launch {model_id}", exact=True)
        launch.wait_for(state="visible", timeout=30_000)
        launch.click()
        self._wait_for_ready_instance(model_id)
        progress.launched = True

        page.reload(wait_until="networkidle")
        page.get_by_role("button", name=f"Chat with {model_id}", exact=True).click()
        page.wait_for_url("**/chat")
        selector = page.get_by_label("Select chat model", exact=True)
        if selector.count():
            selector.select_option(model_id)
        progress.selected = True

        phrase = echo_phrase()
        message = page.get_by_label("Chat message", exact=True)
        message.fill(echo_prompt(phrase))
        page.get_by_role("button", name="Send message", exact=True).click()
        assistant = self._wait_for_assistant(page, expected=phrase)
        # Case-insensitive, matching the waiter. The waiter accepts a
        # case-insensitive match and returns, so a case-sensitive assertion
        # here would reject a response the wait had already declared good.
        text_chat_passed = echo_matched(phrase, assistant)
        progress.text_chat_passed = text_chat_passed
        if not text_chat_passed:
            # Quote the response. A chat failure is otherwise indistinguishable
            # from a hang, and the reason is usually in what the model said.
            progress.text_chat_response = assistant.strip()[:400]

        if vision_contract == "unavailable":
            attach = page.get_by_role("button", name="Attach file", exact=True)
            unavailable = attach.is_disabled()
            progress.false_vision_path_offered = not unavailable
            (
                progress.conversation_persisted_after_reload,
                progress.attachment_persisted_after_reload,
            ) = self._verify_conversation_persistence(
                page,
                expected_user_text=echo_prompt(phrase),
                expected_assistant_text=phrase,
                attachment_name=None,
                assistant_matcher=lambda text: echo_matched(phrase, text),
            )
            return progress.outcome(
                passed=(
                    text_chat_passed
                    and unavailable
                    and progress.conversation_persisted_after_reload
                ),
                message=progress.failure_message(),
            )
        if fixture is None:
            raise ValueError("positive vision browser journey requires a fixture")

        # The vision turn gets its own conversation. Sharing one with the text
        # turn left "repeat this phrase back exactly and say nothing else"
        # standing in context, and a 4B model kept obeying it: shown the
        # fixture, it answered with the previous turn's echo phrase instead of
        # reading the image. That measured instruction carryover, not vision.
        message = self._start_new_conversation(page, model_id=model_id)

        fixture_path = self.artifact_directory / f"{_safe_model_name(model_id)}.png"
        fixture.write(fixture_path)
        captured_before = len(captured_chat_requests)
        page.get_by_label("Image attachment file", exact=True).set_input_files(
            str(fixture_path)
        )
        thumbnail = page.get_by_alt_text(fixture_path.name)
        thumbnail.wait_for(state="visible", timeout=30_000)
        thumbnail_visible = thumbnail.is_visible()
        message.fill(fixture.prompt)
        page.get_by_role("button", name="Send message", exact=True).click()
        retained_attachment = page.get_by_label(
            "User message", exact=True
        ).last.get_by_alt_text(fixture_path.name)
        retained_attachment.wait_for(state="visible", timeout=30_000)
        attachment_retained = retained_attachment.is_visible()
        # Zero, not one: the vision turn is the first exchange of its own
        # conversation, so there is no earlier assistant message to skip past.
        response = self._wait_for_assistant(page, expected=fixture.code)
        code_matched, color_matched, shape_matched = fixture.response_match_details(
            response
        )
        format_matched = fixture.response_format_matches(response)
        attribute_matched = color_matched and shape_matched
        if not (code_matched and attribute_matched and format_matched):
            # Same reasoning as the text chat: without the response text a
            # vision failure cannot be told apart from a broken image path,
            # and the digest match already proves the bytes arrived.
            progress.vision_response = response.strip()[:400]
        vision_requests = captured_chat_requests[captured_before:]
        image_digest = _captured_image_digest(vision_requests)
        evidence = VisionFixtureEvidence(
            channel="dashboard",
            fixture_sha256=fixture.sha256,
            code_sha256=fixture.code_sha256,
            expected_shape=fixture.shape,
            expected_color=fixture.color,
            response_matched_code=code_matched,
            response_matched_attribute=attribute_matched,
            response_matched_color=color_matched,
            response_matched_shape=shape_matched,
            response_matched_format=format_matched,
            request_image_sha256=image_digest,
            thumbnail_visible_before_submit=thumbnail_visible,
            attachment_retained_after_submit=attachment_retained,
            passed=(
                code_matched
                and attribute_matched
                and format_matched
                and image_digest == fixture.sha256
                and thumbnail_visible
                and attachment_retained
            ),
        )
        progress.vision = evidence
        (
            progress.conversation_persisted_after_reload,
            progress.attachment_persisted_after_reload,
        ) = self._verify_conversation_persistence(
            page,
            expected_user_text=fixture.prompt,
            expected_assistant_text=fixture.code,
            attachment_name=fixture_path.name,
        )
        return progress.outcome(
            passed=(
                text_chat_passed
                and evidence.passed
                and progress.conversation_persisted_after_reload
                and progress.attachment_persisted_after_reload is True
            ),
            message=progress.failure_message(),
        )

    def _verify_conversation_persistence(
        self,
        page: Page,
        *,
        expected_user_text: str,
        expected_assistant_text: str,
        attachment_name: str | None,
        assistant_matcher: Callable[[str], bool] | None = None,
    ) -> tuple[bool, bool | None]:
        """Reload the shipped dashboard and require the active thread to survive."""

        page.reload(wait_until="networkidle")
        self._check_abort()
        user_message = (
            page.get_by_label("User message", exact=True).filter(visible=True).last
        )
        assistant_message = (
            page.get_by_label("Assistant message", exact=True).filter(visible=True).last
        )
        user_message.wait_for(state="visible", timeout=30_000)
        assistant_message.wait_for(state="visible", timeout=30_000)
        user_persisted = (
            expected_user_text.casefold() in user_message.inner_text().casefold()
        )
        assistant_text = self._assistant_response_text(assistant_message)
        assistant_persisted = (
            assistant_matcher(assistant_text)
            if assistant_matcher is not None
            else expected_assistant_text.casefold() in assistant_text.casefold()
        )
        if attachment_name is None:
            return user_persisted and assistant_persisted, None
        attachment = user_message.get_by_alt_text(attachment_name)
        return (
            user_persisted and assistant_persisted,
            attachment.count() > 0 and attachment.is_visible(),
        )

    def _dismiss_first_run_consent(self, page: Page) -> bool:
        """Answer the first-run telemetry consent dialog and report whether it appeared.

        A genuinely fresh install shows a modal asking about optional field
        telemetry, and it is a real modal: it covers the page and intercepts
        every pointer event until answered. A long-lived operator browser
        stamped its no-nag marker months ago, so this is invisible on a
        developer fleet and blocks the entire journey on a clean machine.

        "Not now" is the deliberate choice here. It dismisses the dialog and
        leaves fleet consent at ``unasked``, so a throwaway qualification node
        never enables collection and never publishes anything. Answering by
        clicking is also the point: this drives the same control a new user
        does rather than reaching around the UI to seed browser storage.
        """

        dialog = page.get_by_role("dialog").filter(
            has=page.get_by_text("Help make Skulk better?", exact=False)
        )
        try:
            dialog.wait_for(state="visible", timeout=10_000)
        except Exception:  # noqa: BLE001 - absence is a valid, reportable outcome
            # Consent already decided in skulk.yaml, or this browser profile
            # was asked before. Either way there is nothing to answer.
            return False
        dialog.get_by_role("button", name="Not now", exact=True).click()
        dialog.wait_for(state="hidden", timeout=10_000)
        return True

    def _start_new_conversation(self, page: Page, *, model_id: str) -> Locator:
        """Open a fresh conversation and return its message box.

        Each capability check has to be judged on its own answer. Sharing one
        conversation let the text turn's standing instruction ("repeat this
        phrase back exactly and say nothing else") govern the vision turn as
        well, so the model answered an image prompt with the earlier echo
        phrase. That is ordinary multi-turn behavior, not a defect, which is
        exactly why the qualification must not stack the two checks in one
        thread.

        Clicking the control a user would click also keeps this honest: it
        exercises the new-conversation path rather than reaching around the UI
        to reset state.
        """

        page.get_by_role("button", name="+ New", exact=True).click()
        # A new conversation may come up with no model selected, so re-select
        # before asserting anything about the reply.
        selector = page.get_by_label("Select chat model", exact=True)
        if selector.count():
            selector.select_option(model_id)
        message = page.get_by_label("Chat message", exact=True)
        message.wait_for(state="visible", timeout=30_000)
        # The turn only means anything against an empty thread.
        # Conversation history remains mounted in the dashboard while only the
        # selected thread is visible. Counting every labelled assistant card
        # can therefore select a hidden reply from another conversation and
        # judge it against the current turn. Scope the waiter to what a user
        # can actually see in the active thread.
        assistant = page.get_by_label("Assistant message", exact=True).filter(
            visible=True
        )
        deadline = time.monotonic() + 30
        while assistant.count() and time.monotonic() < deadline:
            page.wait_for_timeout(250)
        if assistant.count():
            raise RuntimeError(
                "new conversation still shows prior assistant messages; the "
                "vision turn would inherit the text turn's instructions"
            )
        return message

    def _wait_for_store_model(self, model_id: str) -> None:
        deadline = time.monotonic() + self.model_ready_timeout_s
        with SkulkClient(self.api_base_url) as client:
            while time.monotonic() < deadline:
                self._check_abort()
                registry = client.get_store_registry()
                if registry is not None and _registry_contains(registry, model_id):
                    return
                time.sleep(self.poll_interval_s)
        raise TimeoutError(f"dashboard download did not complete for {model_id}")

    def _wait_for_download_action(
        self,
        page: Page,
        *,
        model_id: str,
    ) -> Locator:
        """Wait for a single-variant, expanded-variant, or added-model action."""

        deadline = time.monotonic() + 60
        download = page.get_by_role("button", name=f"Download {model_id}", exact=True)
        add_and_download = page.get_by_role(
            "button", name=f"Add and download {model_id}", exact=True
        )
        expanded = False
        while time.monotonic() < deadline:
            self._check_abort()
            if download.count() > 0 and download.first.is_visible():
                return download.first
            if add_and_download.count() > 0 and add_and_download.first.is_visible():
                return add_and_download.first
            if not expanded:
                expand = page.get_by_role(
                    "button",
                    name=re.compile(r"^Expand "),
                )
                if expand.count() > 0 and expand.first.is_visible():
                    expand.first.click()
                    expanded = True
            page.wait_for_timeout(250)
        raise TimeoutError(f"dashboard did not find a download action for {model_id}")

    def _wait_for_ready_instance(self, model_id: str) -> None:
        deadline = time.monotonic() + self.model_ready_timeout_s
        with SkulkClient(self.api_base_url) as client:
            while time.monotonic() < deadline:
                self._check_abort()
                placements = client.find_placements_for_model(model_id)
                if any(placement.ready for placement in placements):
                    return
                if any(placement.terminal_failure for placement in placements):
                    raise RuntimeError(f"dashboard placement failed for {model_id}")
                time.sleep(self.poll_interval_s)
        raise TimeoutError(f"dashboard model did not become ready: {model_id}")

    def _wait_for_assistant(
        self,
        page: Page,
        *,
        expected: str,
        after_count: int = 0,
    ) -> str:
        deadline = time.monotonic() + 1800
        # The dashboard keeps hidden conversations mounted. Only visible
        # assistant cards belong to the active thread being qualified.
        assistant = page.get_by_label("Assistant message", exact=True).filter(
            visible=True
        )
        saw_cancel_control = False
        last_text: str | None = None
        stable_without_cancel_polls = 0
        while time.monotonic() < deadline:
            self._check_abort()
            count = assistant.count()
            if count > after_count:
                text = self._assistant_response_text(assistant.nth(count - 1))
                cancel = page.get_by_role(
                    "button", name="Cancel generation", exact=True
                )
                if cancel.count() > 0:
                    saw_cancel_control = True
                    stable_without_cancel_polls = 0
                elif saw_cancel_control:
                    return text
                elif text == last_text:
                    stable_without_cancel_polls += 1
                    if stable_without_cancel_polls >= 1:
                        return text
                else:
                    stable_without_cancel_polls = 0
                last_text = text
            page.wait_for_timeout(500)
        raise TimeoutError(
            "dashboard assistant response did not complete while waiting for "
            f"{expected!r}"
        )

    @staticmethod
    def _assistant_response_text(assistant_message: Locator) -> str:
        """Return model response text without dashboard message chrome when possible."""

        selector = (
            '[data-testid="assistant-response-content"], '
            '[data-testid="streaming-assistant-response-content"]'
        )
        try:
            response = assistant_message.locator(selector).filter(visible=True)
        except AttributeError:
            return assistant_message.inner_text()
        if response.count() > 0:
            return response.last.inner_text()
        return assistant_message.inner_text()

    def _check_abort(self) -> None:
        """Surface a lease or external lifecycle failure during browser waits."""

        if self.abort_check is not None:
            self.abort_check()


def _captured_image_digest(requests: list[dict[str, object]]) -> str | None:
    """Extract and digest the first image URL from captured chat payloads."""

    for request in reversed(requests):
        messages = request.get("messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                if not isinstance(image_url, dict):
                    continue
                url = image_url.get("url")
                if isinstance(url, str):
                    return data_url_sha256(url)
    return None


def _capture_and_close_browser(
    page: Page,
    context: BrowserContext,
    browser: Browser,
    *,
    screenshot_path: Path,
    trace_path: Path,
) -> Exception | None:
    """Retain browser artifacts and always attempt process cleanup."""

    cleanup_error: Exception | None = None
    try:
        page.screenshot(path=str(screenshot_path), full_page=True)
    except Exception as exception:  # noqa: BLE001 - cleanup boundary
        cleanup_error = exception
    try:
        context.tracing.stop(path=str(trace_path))
    except Exception as exception:  # noqa: BLE001 - cleanup boundary
        if cleanup_error is None:
            cleanup_error = exception
    try:
        browser.close()
    except Exception as exception:  # noqa: BLE001 - cleanup boundary
        if cleanup_error is None:
            cleanup_error = exception
    return cleanup_error


def _registry_contains(registry: dict[str, object], model_id: str) -> bool:
    """Recognize a completed model in current and legacy registry shapes."""

    entries = registry.get("models", registry.get("entries"))
    if isinstance(entries, dict):
        return model_id in entries
    if isinstance(entries, list):
        for entry in entries:
            if (
                isinstance(entry, dict)
                and entry.get("model_id", entry.get("id")) == model_id
            ):
                return True
    return False


def _pcm_wav_duration_and_rms(audio: bytes) -> tuple[float, float]:
    """Return duration and signal RMS for an uncompressed mono PCM16 WAV."""

    with wave.open(io.BytesIO(audio), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        compression = reader.getcomptype()
        frames = reader.readframes(frame_count)
    if (
        channels != 1
        or sample_width != 2
        or sample_rate <= 0
        or compression != "NONE"
        or not frames
        or len(frames) % 2 != 0
    ):
        raise ValueError("dashboard speech response must be non-empty mono PCM16 WAV")
    sum_of_squares = 0
    sample_count = len(frames) // 2
    for offset in range(0, len(frames), 2):
        sample = int.from_bytes(frames[offset : offset + 2], "little", signed=True)
        sum_of_squares += sample * sample
    return frame_count / sample_rate, (sum_of_squares / sample_count) ** 0.5


def _fake_microphone_recording_ms(fixture_path: Path) -> int:
    """Record one fixture pass without allowing Chromium to loop the WAV."""

    with wave.open(str(fixture_path), "rb") as reader:
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
    if sample_rate <= 0 or frame_count <= 0:
        raise ValueError("fake microphone fixture must contain audio frames")
    duration_ms = round(frame_count * 1000 / sample_rate)
    return max(1_000, duration_ms - 200)


def _transcript_matches(reference: str, transcript: str) -> bool:
    """Accept a dashboard transcription with at most one quarter word error."""

    reference_words = re.findall(r"\w+(?:['’]\w+)?", reference.casefold())
    transcript_words = re.findall(r"\w+(?:['’]\w+)?", transcript.casefold())
    if not reference_words:
        return not transcript_words
    previous = list(range(len(transcript_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current = [reference_index]
        for transcript_index, transcript_word in enumerate(transcript_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[transcript_index] + 1,
                    previous[transcript_index - 1]
                    + int(reference_word != transcript_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words) <= 0.25


def _safe_model_name(model_id: str) -> str:
    """Convert a public model id into an artifact filename stem."""

    return "".join(character if character.isalnum() else "-" for character in model_id)
