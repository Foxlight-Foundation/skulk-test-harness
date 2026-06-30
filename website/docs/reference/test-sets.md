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
