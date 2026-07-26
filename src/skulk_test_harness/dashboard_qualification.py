"""Live Playwright qualification of the dashboard's real user journey."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Locator, Page, Request, sync_playwright

from skulk_test_harness.client import SkulkClient
from skulk_test_harness.echo_phrase import echo_matched, echo_phrase, echo_prompt
from skulk_test_harness.models import (
    DashboardJourneyOutcome,
    VisionFixtureEvidence,
)
from skulk_test_harness.vision_fixture import VisionFixture, data_url_sha256


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
                page.screenshot(
                    path=str(self.artifact_directory / f"{safe_name}.final.png"),
                    full_page=True,
                )
                context.tracing.stop(
                    path=str(self.artifact_directory / f"{safe_name}.trace.zip")
                )
                browser.close()

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
        launch = page.get_by_role(
            "button", name=f"Launch {model_id}", exact=True
        )
        launch.wait_for(state="visible", timeout=30_000)
        launch.click()
        self._wait_for_ready_instance(model_id)
        progress.launched = True

        page.reload(wait_until="networkidle")
        page.get_by_role(
            "button", name=f"Chat with {model_id}", exact=True
        ).click()
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
            return progress.outcome(
                passed=text_chat_passed and unavailable,
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
        retained_attachment = (
            page.get_by_label("User message", exact=True)
            .last
            .get_by_alt_text(fixture_path.name)
        )
        retained_attachment.wait_for(state="visible", timeout=30_000)
        attachment_retained = retained_attachment.is_visible()
        # Zero, not one: the vision turn is the first exchange of its own
        # conversation, so there is no earlier assistant message to skip past.
        response = self._wait_for_assistant(page, expected=fixture.code)
        code_matched, attribute_matched = fixture.response_matches(response)
        if not (code_matched and attribute_matched):
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
            request_image_sha256=image_digest,
            thumbnail_visible_before_submit=thumbnail_visible,
            attachment_retained_after_submit=attachment_retained,
            passed=(
                code_matched
                and attribute_matched
                and image_digest == fixture.sha256
                and thumbnail_visible
                and attachment_retained
            ),
        )
        progress.vision = evidence
        return progress.outcome(
            passed=text_chat_passed and evidence.passed,
            message=progress.failure_message(),
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
        assistant = page.get_by_label("Assistant message", exact=True)
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
        download = page.get_by_role(
            "button", name=f"Download {model_id}", exact=True
        )
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
        assistant = page.get_by_label("Assistant message", exact=True)
        saw_cancel_control = False
        last_text: str | None = None
        stable_without_cancel_polls = 0
        while time.monotonic() < deadline:
            self._check_abort()
            count = assistant.count()
            if count > after_count:
                text = assistant.nth(count - 1).inner_text()
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


def _registry_contains(registry: dict[str, object], model_id: str) -> bool:
    """Recognize a completed model in current and legacy registry shapes."""

    entries = registry.get("models", registry.get("entries"))
    if isinstance(entries, dict):
        return model_id in entries
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("model_id", entry.get("id")) == model_id:
                return True
    return False


def _safe_model_name(model_id: str) -> str:
    """Convert a public model id into an artifact filename stem."""

    return "".join(character if character.isalnum() else "-" for character in model_id)
