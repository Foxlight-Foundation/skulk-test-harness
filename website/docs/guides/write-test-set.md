---
title: Write A Test Set
---

A test set is a named group of prompts and scoring rules. Each test can be run
against every model selected by a model set.

## Where Test Sets Live

The public file is:

```text
configs/test_sets.yaml
```

Your local config points at it:

```yaml
test_sets_path: configs/test_sets.yaml
```

For private tests, make your own file and point `skulk-harness.yaml` at it:

```yaml
test_sets_path: local/test_sets.yaml
```

## A Tiny Chat Test

```yaml
test_sets:
  my-chat-tests:
    name: my-chat-tests
    description: My first chat checks.
    tests:
      - name: capital-of-france
        kind: chat
        system: You are a careful assistant. Answer directly.
        prompt: What is the capital of France? Answer with only the city name.
        max_tokens: 32
        temperature: 0
        enable_thinking: false
        success:
          min_chars: 5
          required_substrings:
            - Paris
```

The map key and the `name` field must match.

## A Code Test

Code tests use the same request path as chat tests, but the success checks focus
on code-shaped output:

```yaml
test_sets:
  my-code-tests:
    name: my-code-tests
    description: One small code-generation check.
    tests:
      - name: palindrome-function
        kind: code
        prompt: |
          Write a Python function `is_palindrome(text: str) -> bool`.
          Include only one fenced python code block.
        max_tokens: 320
        temperature: 0.1
        enable_thinking: false
        success:
          min_code_block_chars: 120
          required_substrings:
            - is_palindrome
            - bool
```

## A Tool-Calling Test

Tool tests let you check OpenAI-style function calls without needing a real
weather service, calculator, database, or monitoring API. The harness returns
static mock results.

```yaml
test_sets:
  my-tool-tests:
    name: my-tool-tests
    description: Tool routing with one mock result.
    tests:
      - name: forced-weather-tool-call
        kind: tool
        system: Use tools whenever the answer depends on live data.
        prompt: What is the current weather in Cedar Rapids, Iowa?
        max_tokens: 240
        temperature: 0
        enable_thinking: false
        tools:
          - type: function
            function:
              name: get_weather
              description: Get current weather for a city.
              parameters:
                type: object
                properties:
                  location:
                    type: string
                  units:
                    type: string
                    enum:
                      - fahrenheit
                      - celsius
                required:
                  - location
                  - units
        tool_choice:
          type: function
          function:
            name: get_weather
        parallel_tool_calls: false
        tool_mocks:
          - name: get_weather
            content: '{"location":"Cedar Rapids, Iowa","temperature_f":72,"condition":"clear"}'
        success:
          min_chars: 20
          min_tool_calls: 1
          required_substrings:
            - "72"
            - clear
          expected_tool_calls:
            - name: get_weather
              required_arguments:
                - location
              argument_substrings:
                location: Cedar Rapids
```

## A Cancellation Test

Cancellation tests close a stream after a few chunks and then send a follow-up
prompt to verify the serving path is still healthy:

```yaml
test_sets:
  my-cancel-tests:
    name: my-cancel-tests
    description: Cancel a stream and verify follow-up health.
    tests:
      - name: cancel-mid-stream-and-recover
        kind: cancel
        system: You are a verbose technical writer.
        prompt: Keep writing about distributed inference until stopped.
        max_tokens: 512
        temperature: 0.7
        enable_thinking: false
        cancel_after_chunks: 3
        followup_prompt: Reply with exactly HEALTHY and nothing else.
        success:
          min_chars: 7
          required_substrings:
            - HEALTHY
```

## An Embedding Test

Embedding tests call the embeddings endpoint and check vector shape:

```yaml
test_sets:
  my-embedding-tests:
    name: my-embedding-tests
    description: One embedding vector check.
    tests:
      - name: sentence-vector-shape
        kind: embedding
        prompt: Skulk embeddings smoke test.
        embedding_input: Skulk embeddings smoke test.
        expected_embedding_dimensions: 384
        min_embedding_norm: 0.01
        success:
          min_chars: 0
```

## Speech Tests

Speech synthesis tests call `/v1/audio/speech`, score the binary response, and
persist the generated audio under the run's `artifacts/` directory:

```yaml
test_sets:
  my-speech-tests:
    name: my-speech-tests
    description: One text-to-speech check.
    tests:
      - name: tts-wav-audio-bytes
        kind: audio_speech
        prompt: Hello world from Skulk.
        audio_response_format: wav
        success:
          min_chars: 0
          min_audio_bytes: 1024
```

Streaming speech synthesis tests call `/v1/audio/speech` with `stream=true`,
score the final audio bytes, and save a `.stream.json` timing sidecar next to
the generated audio. Use `min_stream_span_s` when the suite should reject
responses that are only split into chunks after synthesis has already finished:

```yaml
test_sets:
  my-streaming-speech-tests:
    name: my-streaming-speech-tests
    description: One text-to-speech streaming check.
    tests:
      - name: tts-mp3-streaming-chunks
        kind: audio_speech_streaming
        prompt: Hello world from Skulk streaming speech.
        audio_response_format: mp3
        speech_streaming_interval: 0.25
        success:
          min_chars: 0
          min_audio_bytes: 1024
          min_stream_chunks: 2
          min_stream_span_s: 0.5
```

Pressure tests distribute independent streaming clients across API owners
discovered from `/v1/diagnostics/cluster`. Every request is scored and saved as
audio plus a timing sidecar. Slow-reader workers exercise isolation without
making the whole suite destructive:

```yaml
test_sets:
  my-speech-pressure:
    name: my-speech-pressure
    description: Concurrent multi-owner TTS streaming.
    tests:
      - name: tts-multi-owner-pressure
        kind: audio_speech_pressure
        prompt: Skulk progressive speech pressure.
        audio_response_format: mp3
        speech_concurrency: 6
        speech_requests_per_worker: 1
        speech_owner_count: 3
        speech_slow_workers: 1
        speech_slow_reader_delay_s: 0.25
        success:
          min_chars: 0
          min_audio_bytes: 1024
          min_stream_chunks: 2
          min_stream_span_s: 0.25
```

Roundtrip tests use a mounted TTS model as the primary target, persist that
generated audio, then place a speech-to-text model through the normal Skulk
store-backed lifecycle and transcribe the generated audio:

```yaml
test_sets:
  my-speech-roundtrip:
    name: my-speech-roundtrip
    description: TTS output transcribed by an STT model.
    tests:
      - name: tts-to-stt-hello-world
        kind: speech_roundtrip
        prompt: Hello world.
        audio_response_format: wav
        transcription_response_format: json
        transcription_language: en
        success:
          min_chars: 5
          min_audio_bytes: 1024
          required_substrings:
            - hello
```

## Success Criteria

| Field | Use it when you need to check... |
| --- | --- |
| `min_chars` | visible output length |
| `min_code_block_chars` | fenced code length |
| `min_list_items` | list structure |
| `min_generated_chars` | visible content plus separated reasoning |
| `min_tool_calls` | number of tool calls |
| `in_order_integers` | ordered integer streaming coherence |
| `required_substrings` | required facts or tokens |
| `forbidden_substrings` | strings that must not appear |
| `required_regexes` | structure or flexible patterns |
| `expected_tool_calls` | function names and argument checks |
| `require_logprobs` | per-token logprob availability |
| `min_reasoning_chars` | separated reasoning output |
| `min_wall_tps` | a throughput floor for benchmarks |
| `min_audio_bytes` | encoded TTS audio size |

## Check Your Set

```bash
uv run skulk-harness tests sets --config skulk-harness.yaml
uv run skulk-harness plan --model-set store-smoke --test-set my-chat-tests
```

Start with one small test. Add more only after the first one passes and the
report is easy to understand.
