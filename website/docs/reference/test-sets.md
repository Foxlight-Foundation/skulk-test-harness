---
title: Test Sets
---

Test sets live in a YAML file with one top-level key: `test_sets`.

## File Shape

```yaml
test_sets:
  chat-tests:
    name: chat-tests
    description: General chat sanity and instruction-following tests.
    tests:
      - name: concise-factual-answer
        kind: chat
        prompt: What is the capital of France?
        max_tokens: 64
        temperature: 0
        success:
          min_chars: 5
          required_substrings:
            - Paris
```

Each map key must match the test set's `name`.

## Test Kinds

| Kind | Purpose |
| --- | --- |
| `chat` | General chat completion behavior |
| `code` | Code generation with code-shaped checks |
| `artifact` | Artifact-style generation checks |
| `tool` | OpenAI-style tool call behavior |
| `cancel` | Streaming cancellation and follow-up health |
| `error` | Expected API error behavior |
| `embedding` | Embeddings endpoint behavior |
| `audio_speech` | Text-to-speech endpoint behavior; generated audio is saved as an artifact |
| `audio_speech_streaming` | Stable card-qualified text-to-speech streaming behavior; generated audio and timing sidecar are saved as artifacts |
| `audio_speech_pressure` | Concurrent streaming TTS across discovered API owners, with deterministic local/remote routing, DATA diagnostics, optional chat workers, and one audio/timing artifact per speech request |
| `audio_voices` | Static voice-catalog behavior for a mounted TTS model |
| `audio_transcription` | Speech-to-text endpoint behavior with an audio fixture |
| `audio_transcription_streaming` | Uploaded-audio SSE transcript deltas, terminal lifecycle, early-close cancellation, and saved input/timeline artifacts |
| `realtime_transcription` | Semantic TTS-to-realtime-STT WebSocket roundtrip, disconnect recovery, local/remote ownership, and provider diagnostics |
| `realtime_conversation` | Persistent server-VAD voice loop with automatic commits, multiple turns, assistant text/audio, optional barge-in, local/remote ownership, and saved event evidence |
| `fabric_speech_chain` | Explicit Fabric STT-to-chat-to-TTS composition with transcript, assistant text, response audio, cancellation, local/remote ownership, and provider diagnostics |
| `speech_roundtrip` | TTS output saved as an artifact and piped into a mounted STT model |
| `speech_translation_roundtrip` | TTS output saved and translated to English by a mounted translation model |
| `speech_reference_roundtrip` | A donor TTS clip conditions a multipart TTS request; both clips are saved |

## Prompt Test Fields

| Field | Meaning |
| --- | --- |
| `name` | Test name used in reports |
| `kind` | One of the supported test kinds |
| `description` | Optional human explanation |
| `system` | Optional system message |
| `prompt` | Main user prompt |
| `max_tokens` | Output token budget |
| `temperature` | Sampling temperature |
| `top_p` | Optional nucleus sampling value |
| `enable_thinking` | Optional reasoning toggle |
| `reasoning_effort` | Optional reasoning effort value |
| `prompt_repetitions` | Repeat prompt text before sending |
| `images` | OpenAI-style image input parts |
| `tools` | OpenAI-style tool schemas |
| `tool_choice` | Tool choice policy |
| `parallel_tool_calls` | Whether parallel tool calls are allowed |
| `tool_mocks` | Static tool results |
| `cancel_after_chunks` | Stream chunks before cancellation |
| `followup_prompt` | Health check after cancel or expected error |
| `expected_error_statuses` | Acceptable statuses for `kind: error` |
| `expected_error_substrings` | Required text in an expected error |
| `embedding_input` | Embedding request input |
| `expected_embedding_dimensions` | Required vector dimensionality |
| `min_embedding_norm` | Minimum L2 norm for embedding vectors |
| `audio_response_format` | TTS audio response format, such as `wav` |
| `speech_voice` | Optional voice name for TTS |
| `speech_speed` | Optional speech speed multiplier for TTS |
| `reference_model_id` | Donor TTS model for `kind: speech_reference_roundtrip` |
| `reference_text` | Transcript spoken in the generated reference clip; defaults to `prompt` |
| `expected_voice_ids` | Voice identifiers required for `kind: audio_voices` |
| `speech_streaming_interval` | Optional `streaming_interval` hint for `kind: audio_speech_streaming` |
| `speech_concurrency` | Concurrent workers for `kind: audio_speech_pressure` |
| `speech_requests_per_worker` | Sequential requests issued by each pressure worker |
| `speech_owner_count` | Distinct reachable API owners selected from cluster diagnostics |
| `speech_owner_topology` | `any` or `local_remote`; the latter selects one owner on the TTS serving node and the rest away from it |
| `speech_assert_data_plane_diagnostics` | Require an idle pre-test baseline, lifecycle/egress counters that cover successful requests, zero live gauges after the workload, and no new anomalies (including idle stream reclamation); saves a sanitized diagnostics sidecar |
| `speech_chat_model_id` | Optional secondary text model mounted for mixed chat-plus-TTS pressure |
| `speech_chat_concurrency` | Concurrent streaming chat workers run beside speech pressure |
| `speech_chat_prompt` | Prompt sent by mixed-pressure chat workers |
| `speech_slow_workers` | Leading pressure workers that intentionally delay stream reads |
| `speech_slow_reader_delay_s` | Delay after each received chunk for slow pressure workers |
| `input_audio_path` | Optional local fixture path for batch or streaming audio transcription; streaming tests can generate a TTS fixture instead |
| `input_audio_mime_type` | Optional MIME type for transcription fixture upload; inferred from the fixture extension when omitted |
| `transcription_model_id` | Optional STT model used by `kind: speech_roundtrip` |
| `speech_synthesis_model_id` | Optional TTS fixture model used by realtime or uploaded-audio streaming transcription |
| `realtime_response_model_id` | Mounted chat participant required by conversational realtime and Fabric-chain tests |
| `realtime_response_tts_model_id` | Mounted response TTS participant required by conversational realtime and Fabric-chain tests |
| `transcription_response_format` | STT response format, such as `json` or `text` |
| `transcription_language` | Optional STT language hint |
| `transcription_cancel_after_deltas` | Close a secondary uploaded-audio stream after this many transcript deltas; zero disables the probe |
| `realtime_frame_duration_ms` | PCM16 append-frame duration; defaults to the dashboard's 100 ms |
| `realtime_pace_audio` | Send frames at media cadence instead of bursting them |
| `realtime_cancel_after_frames` | Run a disconnect probe after this many uncommitted frames |
| `realtime_assert_provider_diagnostics` | Require lifecycle/media counters and drained realtime provider gauges |
| `realtime_server_vad` | Enable server VAD and automatic input commits |
| `realtime_turn_count` | Number of utterances sent over one persistent socket, from 1 through 4 |
| `realtime_barge_in` | Send the next turn after response audio begins and require cancellation of the superseded response |
| `top_logprobs` | Request ranked logprob alternatives |
| `repetitions` | Number of times to repeat the test |
| `success` | Scoring rules |

## Success Criteria

| Field | Meaning |
| --- | --- |
| `min_chars` | Minimum visible output characters |
| `min_code_block_chars` | Minimum fenced code block characters |
| `min_list_items` | Minimum structured list items |
| `min_generated_chars` | Minimum visible plus reasoning characters |
| `min_tool_calls` | Minimum tool call count |
| `in_order_integers` | Require integers 1 through N in order |
| `required_substrings` | Strings that must appear |
| `forbidden_substrings` | Strings that must not appear |
| `required_regexes` | Regex patterns that must match |
| `expected_tool_calls` | Expected function names and arguments |
| `require_html_artifact` | Require an HTML artifact |
| `require_logprobs` | Require token logprobs in streamed output |
| `min_reasoning_chars` | Minimum separated reasoning characters |
| `forbid_in_reasoning` | Apply forbidden strings to reasoning too |
| `min_wall_tps` | Minimum wall-clock decode tokens per second |
| `min_audio_bytes` | Minimum encoded audio bytes for speech synthesis |
| `min_stream_chunks` | Minimum streamed response chunk count |
| `max_first_byte_s` | Optional maximum time to first streamed byte/token |
| `min_stream_span_s` | Minimum elapsed seconds between first and last streamed chunks |
| `min_transcript_deltas` | Minimum incremental transcript events before the realtime final transcript |

## Public Built-In Sets

| Name | Purpose |
| --- | --- |
| `chat-tests` | General chat sanity |
| `code-tests` | Code generation checks |
| `tool-tests` | Tool-call coverage with static mocks |
| `throughput` | Sustained decode throughput smoke |
| `cancellation` | Streaming cancellation coverage |
| `context-admission` | Oversized request guard |
| `embeddings` | Embeddings endpoint coverage |
| `speech-synthesis` | Text-to-speech endpoint coverage |
| `speech-synthesis-streaming` | Experimental text-to-speech streaming coverage |
| `speech-data-pressure` | Concurrent local/remote TTS pressure with DATA diagnostics |
| `speech-roundtrip` | TTS-to-STT endpoint coverage |
| `realtime-transcription` | Realtime WebSocket STT, semantic transcript, cancellation, and provider diagnostics |
| `vision` | Multimodal image input coverage |
| `served-speculation` | Served speculation correctness and throughput |

## Validation

List sets after editing:

```bash
uv run skulk-harness tests sets --config skulk-harness.yaml
```

Plan one set before executing it:

```bash
uv run skulk-harness plan --model-set store-smoke --test-set my-test-set
```
