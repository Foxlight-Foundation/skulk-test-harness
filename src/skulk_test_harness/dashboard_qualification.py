"""Live Playwright qualification of the dashboard's real user journey."""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Callable
from pathlib import Path

from playwright.sync_api import Locator, Page, Request, sync_playwright

from skulk_test_harness.client import SkulkClient
from skulk_test_harness.models import (
    DashboardJourneyOutcome,
    VisionFixtureEvidence,
)
from skulk_test_harness.vision_fixture import VisionFixture, data_url_sha256


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
            outcome = DashboardJourneyOutcome(model_id=model_id)
            try:
                outcome = self._run_journey(
                    page,
                    model_id=model_id,
                    vision_contract=vision_contract,
                    fixture=fixture,
                    captured_chat_requests=captured_chat_requests,
                )
                return outcome
            except Exception as exception:  # noqa: BLE001 - report browser boundary
                return outcome.model_copy(
                    update={"passed": False, "message": str(exception)}
                )
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
    ) -> DashboardJourneyOutcome:
        page.goto(f"{self.api_base_url}/model-store", wait_until="networkidle")
        self._check_abort()
        page.get_by_role("button", name="Find Models", exact=True).click()
        search = page.get_by_label("Search models", exact=True)
        search.fill(model_id)
        download = self._wait_for_download_action(page, model_id=model_id)
        found = True
        download.click()
        download_started = True
        page.get_by_role("button", name="Close", exact=True).click()
        self._wait_for_store_model(model_id)

        page.reload(wait_until="networkidle")
        launch = page.get_by_role(
            "button", name=f"Launch {model_id}", exact=True
        )
        launch.wait_for(state="visible", timeout=30_000)
        launch.click()
        self._wait_for_ready_instance(model_id)
        launched = True

        page.reload(wait_until="networkidle")
        page.get_by_role(
            "button", name=f"Chat with {model_id}", exact=True
        ).click()
        page.wait_for_url("**/chat")
        selector = page.get_by_label("Select chat model", exact=True)
        if selector.count():
            selector.select_option(model_id)
        selected = True

        token = f"FRESH-{secrets.token_hex(4).upper()}"
        prompt = f"Reply with this token exactly once and nothing else: {token}"
        message = page.get_by_label("Chat message", exact=True)
        message.fill(prompt)
        page.get_by_role("button", name="Send message", exact=True).click()
        assistant = self._wait_for_assistant(page, expected=token)
        text_chat_passed = token in assistant

        if vision_contract == "unavailable":
            attach = page.get_by_role("button", name="Attach file", exact=True)
            unavailable = attach.is_disabled()
            return DashboardJourneyOutcome(
                model_id=model_id,
                found=found,
                download_started=download_started,
                launched=launched,
                selected=selected,
                text_chat_passed=text_chat_passed,
                false_vision_path_offered=not unavailable,
                passed=text_chat_passed and unavailable,
            )
        if fixture is None:
            raise ValueError("positive vision browser journey requires a fixture")

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
        response = self._wait_for_assistant(
            page,
            expected=fixture.code,
            after_count=1,
        )
        code_matched, attribute_matched = fixture.response_matches(response)
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
        return DashboardJourneyOutcome(
            model_id=model_id,
            found=found,
            download_started=download_started,
            launched=launched,
            selected=selected,
            text_chat_passed=text_chat_passed,
            vision=evidence,
            passed=text_chat_passed and evidence.passed,
        )

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
        while time.monotonic() < deadline:
            self._check_abort()
            count = assistant.count()
            if count > after_count:
                text = assistant.nth(count - 1).inner_text()
                if expected.upper() in text.upper():
                    return text
                cancel = page.get_by_role(
                    "button", name="Cancel generation", exact=True
                )
                if cancel.count() == 0:
                    return text
            page.wait_for_timeout(500)
        raise TimeoutError("dashboard assistant response did not complete")

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
