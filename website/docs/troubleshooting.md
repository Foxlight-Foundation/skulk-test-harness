---
title: Troubleshooting
---

This page starts with the most common first-run problems.

## Quick Triage

| Symptom | First thing to check |
| --- | --- |
| `models sets` fails | YAML syntax or schema error |
| `tests sets` fails | YAML syntax or schema error |
| `doctor` cannot connect | `api_base_url` and whether Skulk is running |
| No models selected | Model set selector and Skulk store contents |
| Placement fails | Cluster capacity, model compatibility, runner health |
| Test fails but model answered | Success criteria may be too strict or wrong for the model |
| Tool test fails | Backend or model may not support tool calls |
| Report directory missing | Check `output_dir` and command exit output |

## Config Does Not Load

Run the smallest local command:

```bash
uv run skulk-harness models sets --config skulk-harness.example.yaml
```

If that works, your install is fine and your private config or YAML files need
attention.

Common YAML problems:

| Problem | Example |
| --- | --- |
| Wrong indentation | A list item is aligned under the wrong parent |
| Key and name mismatch | `my-set:` but `name: other-set` |
| Unknown field | A typo such as `max_token` instead of `max_tokens` |
| Wrong type | `max_models: many` instead of an integer |

## The Cluster Is Down

`doctor` needs a live Skulk API:

```bash
uv run skulk-harness doctor
```

If it fails:

1. Confirm Skulk is running.
2. Confirm the API port.
3. Confirm you can reach the host from this machine.
4. Update `api_base_url`.
5. Try `doctor` again.

The harness does not boot the cluster for you.

## No Models Are Selected

First list the store:

```bash
uv run skulk-harness models store
```

Then list the catalog:

```bash
uv run skulk-harness models catalog
```

If the model set uses `source: store`, the model must be in the store. If it
uses `source: catalog`, it must appear in the catalog. If it uses `source:
both`, either source can match.

## Placement Does Not Become Ready

Placement can fail for several reasons:

| Cause | What to inspect |
| --- | --- |
| Not enough memory | Skulk state and node memory |
| Wrong backend | Model card backend compatibility |
| Runner startup failure | Skulk runner logs |
| Model not downloaded | Model store status |
| Previous run still active | Existing instances and cleanup state |

Start with:

```bash
uv run skulk-harness doctor
uv run skulk-harness models store
```

## Tool Tests Fail

Tool tests need both a model that can emit tool calls and a serving path that
preserves them.

Try:

1. Run `chat-tests` first to prove basic generation works.
2. Use `tool_choice` to force one known tool call.
3. Keep the tool schema small.
4. Check `report.json` for `tool_calls`.

## Stability Command Refuses To Run

This is expected:

```text
Refusing destructive stability command.
Pass --execute-destructive to allow SSH kill/relaunch operations.
```

Add the flag only after reviewing your config:

```bash
uv run skulk-harness stability failover --execute-destructive
```

For destructive suites, also confirm `cluster_nodes` contains the right
`ssh_host`, `kill_command`, and `relaunch_command` for every node the suite may
touch.

## Report Says Failed

A failed report is useful. Do not rerun immediately. First save the report path
from the CLI output and inspect:

```bash
ls runs/<run-id>
sed -n '1,220p' runs/<run-id>/summary.md
```

Then decide whether the failure is:

| Failure kind | Meaning |
| --- | --- |
| Harness or config issue | The test did not run as intended |
| Cluster issue | Skulk could not place, serve, stream, or recover |
| Model behavior issue | The model output did not satisfy the test |
| Test design issue | The success criteria were too strict, too vague, or wrong |

That distinction matters. Fixing the wrong layer wastes time.
