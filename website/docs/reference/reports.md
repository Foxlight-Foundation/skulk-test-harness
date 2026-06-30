---
title: Reports
---

Every normal harness plan or run writes a report directory under `output_dir`.
The default output directory is `runs/`.

## Directory Shape

```text
runs/<run-id>/
  report.json
  events.jsonl
  summary.md
```

Stability suites write:

```text
runs/<run-id>/
  report.json
  summary.md
```

## Files

| File | Audience | Purpose |
| --- | --- | --- |
| `summary.md` | Humans | Fast reading and pull request comments |
| `report.json` | Automation | Full structured report |
| `events.jsonl` | Scripts and logs | One event per line |

## Summary Sections

| Section | Meaning |
| --- | --- |
| Models | Resolved model ids and how they were selected |
| Placements | Instances, nodes, reuse, and readiness |
| Results | Pass or fail for each model, test, and repetition |
| Issues | Run-level and result-level problems |

## Important Metrics

| Metric | Meaning |
| --- | --- |
| `ttft_s` | Time to first token in seconds |
| `wall_tps` | Approximate decode tokens per second from wall-clock timing |
| `skulk_prompt_tps` | Prompt throughput reported by Skulk when available |
| `skulk_generation_tps` | Generation throughput reported by Skulk when available |
| `output_chars` | Visible output character count |
| `generated_chars` | Visible plus separated reasoning character count |
| `chunks` | Number of streamed chunks observed |

## A Small Result Example

```json
{
  "model_id": "mlx-community/Qwen3.5-9B-4bit",
  "test_name": "concise-factual-answer",
  "repetition": 1,
  "passed": true,
  "output_text": "Paris",
  "metrics": {
    "elapsed_s": 1.42,
    "ttft_s": 0.81,
    "output_chars": 5,
    "generated_chars": 5,
    "chunks": 2,
    "wall_tps": 18.0
  },
  "issues": []
}
```

## How To Read Failures

Use this order:

1. Open `summary.md`.
2. Read `Issues`.
3. Find the failing result row.
4. Open `report.json` if the summary is not enough.
5. Compare the test's `success` rules with `output_text`, `reasoning_text`, and
   `tool_calls`.

Common failure patterns:

| Symptom | Likely next step |
| --- | --- |
| No models selected | Check the model set selector and model store |
| Placement not ready | Check cluster capacity and runner logs |
| Required substring missing | Check whether the prompt is too open-ended |
| Tool call missing | Confirm the model and backend support tool calls |
| Throughput below floor | Compare with a recent known-good benchmark |
| Context error expected but absent | Check Skulk context admission behavior |
