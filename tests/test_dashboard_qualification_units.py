import io
import wave

import pytest

from skulk_test_harness.dashboard_qualification import (
    _assert_dashboard_speech_request,
    _assert_pinned_segment_requests,
    _captured_pcm_sample_rate,
    _captured_speech_request_body,
    _pcm16_wav_bytes,
    _pcm_wav_duration_and_rms,
    _raw_pcm16_duration_and_rms,
)


def _capture(*, text: str, voice: str = "angus") -> dict[str, object]:
    """Build one browser-shaped speech request capture."""

    return {
        "requestBody": {
            "model": "org/TTS",
            "input": text,
            "voice": voice,
            "lang_code": "English",
        }
    }


def test_captured_dashboard_request_requires_model_voice_and_language() -> None:
    body = _captured_speech_request_body(_capture(text="First sentence."))

    _assert_dashboard_speech_request(
        body,
        model_id="org/TTS",
        expected_voice="angus",
        expected_language="English",
    )

    with pytest.raises(ValueError, match="wrong voice"):
        _assert_dashboard_speech_request(
            body,
            model_id="org/TTS",
            expected_voice="ember",
            expected_language="English",
        )

    body_without_voice = dict(body)
    del body_without_voice["voice"]
    _assert_dashboard_speech_request(
        body_without_voice,
        model_id="org/TTS",
        expected_voice=None,
        expected_language="English",
    )


def test_pinned_segments_require_distinct_inputs_and_one_voice() -> None:
    assert _assert_pinned_segment_requests(
        [
            _capture(text="First sentence."),
            _capture(text="Second sentence."),
        ],
        model_id="org/TTS",
        expected_voice="angus",
        expected_language="English",
    ) == (2, True, True)

    with pytest.raises(RuntimeError, match="changed voices"):
        _assert_pinned_segment_requests(
            [
                _capture(text="First sentence."),
                _capture(text="Second sentence.", voice="ember"),
            ],
            model_id="org/TTS",
            expected_voice=None,
            expected_language="English",
        )


def test_raw_pcm_capture_becomes_a_playable_wav() -> None:
    samples = (1000).to_bytes(2, "little", signed=True) * 24_000
    capture = {
        "sampleRate": "24000",
        "channels": "1",
        "sampleFormat": "s16le",
    }

    sample_rate = _captured_pcm_sample_rate(capture)
    duration, rms = _raw_pcm16_duration_and_rms(samples, sample_rate)
    wav_bytes = _pcm16_wav_bytes(samples, sample_rate)

    assert duration == pytest.approx(1.0)
    assert rms == pytest.approx(1000.0)
    assert _pcm_wav_duration_and_rms(wav_bytes) == pytest.approx((1.0, 1000.0))
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        assert reader.getframerate() == 24_000
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
