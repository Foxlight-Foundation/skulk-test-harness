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
| `submit` fails with no token | GitHub auth: `gh auth login` or `GH_TOKEN` |
| `submit` rejected with 422 | The run fails a ledger validation gate |
| `compare` warns `missing fingerprint` | One side is an old run without a fingerprint |

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

## Submit Says No GitHub Token

```text
submit failed: no GitHub token: pass --github-token, set GH_TOKEN, or log in with the gh CLI
```

`submit` needs a GitHub token to attribute the submission. It looks, in
order, at `--github-token`, `GH_TOKEN`, `GITHUB_TOKEN`, and finally the `gh`
CLI. The simplest fix:

```bash
gh auth login
```

or export a token:

```bash
export GH_TOKEN=<your token>
```

A `401` from the ingest means a token was found but rejected; check that it
is valid and not expired.

## Submit Rejected With 422

A `422` means the payload failed one of the ledger's validation gates. The
error message names the gate. The common ones:

| Gate | Meaning | Fix |
| --- | --- | --- |
| Run id shape | `run_id` is not harness-shaped | Submit an unmodified harness `report.json` |
| Results present | The report has no test results | You submitted a dry-run plan; rerun with `--execute` |
| Fingerprint with nodes | No fingerprint, or no nodes in it | Rerun on a current harness against a reachable cluster |
| Schema 2.x | The fingerprint schema is too old | Update the harness and rerun |

A `409` is different and benign: the ledger already has this exact run.

See [Submit to the ledger](guides/submit-to-the-ledger.md) for the full
submission flow and what gets redacted.

## Compare Warns About A Missing Fingerprint

`compare` prints a `missing fingerprint` guard when at least one side has no
fingerprint block in its `report.json`. This is normal for runs recorded by
older harness versions: there is nothing to verify the cluster, Skulk
version, or cache conditions against, so the delta is indicative rather than
conclusive.

If the baseline matters, reproduce it on the current harness so both sides
carry fingerprints. The other guards are explained in
[Compare runs](guides/compare-runs.md).

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
