---
title: What E2E Testing Means
---

End-to-end testing means testing the system through the same outside surface a
real user or tool uses. For Skulk, that means sending API requests to a real
Skulk cluster and checking the result.

## Test Types Compared

| Test type | What it checks | Example | Needs a cluster |
| --- | --- | --- | --- |
| Unit | One small function or class | YAML parsing rejects an unknown field | No |
| Integration | A few pieces together | The CLI loads config and model sets | Usually no |
| End-to-end | The real product path | Place a model and send chat requests | Yes |
| Stability | Operational behavior during stress or failure | Kill a node and verify recovery | Yes |

The harness has two E2E environments. An attached run proves behavior on the
configuration already present on a fleet. A fresh-install run proves the
installer, generated configuration, served dashboard, and model journey a new
user actually receives. Attached coverage cannot satisfy the release gate.

## Why LLM E2E Tests Are Different

Traditional services often return exact values. Language models can return many
valid answers. A good harness test usually checks properties rather than exact
full text.

| Better than exact text | Why |
| --- | --- |
| `required_substrings` | Checks a key fact without requiring identical wording |
| `required_regexes` | Checks structure such as a numbered list |
| `min_list_items` | Checks formatting without caring about specific bullets |
| `expected_tool_calls` | Checks tool routing and argument shape |
| `min_wall_tps` | Checks performance when a benchmark needs a floor |

For example, this is too brittle:

```yaml
success:
  required_substrings:
    - "Paris is the capital city of France."
```

This is usually better:

```yaml
success:
  min_chars: 5
  required_substrings:
    - Paris
```

## The Three Safety Levels

| Level | Command shape | What happens |
| --- | --- | --- |
| Read local YAML | `models sets`, `tests sets` | No cluster calls |
| Plan | `plan` or `run` without `--execute` | Reads cluster state and writes a plan |
| Execute | `run --execute` | Can place models and send live requests |
| Fresh install | `fresh-install qualify` | Installs into an empty HOME, tests, then restores or deletes the target |

Destructive stability commands have an extra safety level. `failover`, `churn`,
and `refusal` require `--execute-destructive` before any SSH or destructive API
side effect can happen.

## A Good First Test

A good first e2e test should be:

| Quality | What it looks like |
| --- | --- |
| Small | One prompt, low `max_tokens`, one repetition |
| Deterministic | `temperature: 0` or a low value |
| Easy to explain | A human can tell why it passed |
| Cheap | It does not require a large model or long generation |
| Safe | It does not require destructive cluster actions |

The built-in `chat-tests` set is designed for that kind of first pass.

Vision release qualification is intentionally exact rather than subjective.
Each API and dashboard request gets a different generated PNG containing a
random hidden code plus a random colored shape. The response must contain the
exact code and visual attributes. The dashboard request body is captured and
its image data URL must decode to the uploaded fixture's digest.
