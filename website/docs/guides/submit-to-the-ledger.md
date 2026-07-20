---
title: Submit To The Ledger
---

The [Skulk benchmarks ledger](https://benchmarks.foxlight.ai) is a public
collection of harness runs from the community. It exists so "what does Skulk
do on my hardware?" has answers backed by real reports instead of anecdotes.

It is a **ledger, not a leaderboard**. Every community run is badged with the
GitHub account that submitted it and the hardware that produced it. Community
numbers are never blended into the site's headline numbers; they stand next
to them, labeled as what they are.

This guide takes a finished run from your `runs/` directory to the ledger.

## What you need

| Requirement | Why |
| --- | --- |
| A completed execute run under `runs/` | Only real runs are submittable; dry-run plans have no results |
| A GitHub account | Submissions are attributed to the submitter |
| A GitHub token | Authenticates the submission; see below |

The harness finds a token in this order:

1. `--github-token <token>` on the command line
2. The `GH_TOKEN` environment variable
3. The `GITHUB_TOKEN` environment variable
4. The [`gh` CLI](https://cli.github.com/), if you are logged in
   (`gh auth login`); the harness runs `gh auth token` for you

If you already use `gh`, there is nothing to set up.

`submit` never contacts your Skulk cluster. It reads a local report file and
talks only to the ledger's ingest API. By default that is the public
Foxlight ledger; a self-hosted or staging ingest can be targeted with
`--ingest-url` or the `SKULK_INGEST_URL` environment variable (the flag
wins when both are set).

## Step 1: inspect what would be sent

Always look before you send. `--dry-run` prints the exact payload that would
leave your machine, and sends nothing:

```bash
uv run skulk-harness submit runs/<your-run> --dry-run
```

You can pass either the run directory or its `report.json` directly. Read the
output: it is your run's `report.json` after redaction. If anything in it
surprises you, do not submit it.

## What is redacted, exactly

Redaction happens **on your machine, before anything is sent**. The dry-run
payload is byte-for-byte what the ingest receives. The following never leave
your machine:

| Removed | Why |
| --- | --- |
| Generated text (`output_text`, `reasoning_text`, `tool_calls`) | The ledger publishes performance numbers, not model output |
| Local artifact paths | They describe your filesystem |
| Issue evidence (run-level and result-level) | Evidence blobs can embed generated content; the ledger only renders an issue's severity and message |
| Operator notes | Free-form text you wrote for yourself |
| Run names, including custom labels inside the run id | A `--run-name` can carry a host, lab, or customer name; a labeled run id is replaced with a deterministic hash of the original (so resubmitting the same run still deduplicates), while default-shaped run ids like `20260709-141530-store-smoke-chat-tests` are kept as-is. Concretely: a run named `lab7-perf-check` submits as `20260709-141530-submitted-3f9c2a81d0`, while a default spec-derived name is left untouched |
| Repository paths and branch names | Paths describe your machine; branch names can carry labels. The commit hash stays, because it is the precise code provenance |
| The cluster API URL | It is your network address |
| Node friendly names and the topology label | Names you gave your machines are identity, not hardware |

What is **kept**: the metrics, pass/fail results, the run spec (model set and
test set names), and the fingerprint's hardware facts (RAM, accelerator
vendor and name, Skulk version, cache state).

Node ids are also kept in the payload: the ledger joins each placement's
`node_ids` to the fingerprint's node entries so a result is attributed to the
exact machines that served it. Node ids are random identifiers with no
personal content, and the ledger hashes them server-side before anything is
published.

One class of report is refused outright: **stability-suite reports** (from
`stability failover`, `churn`, `soak`, `refusal`). Their observations embed
cluster-specific detail, and the ledger does not accept them. The refusal
happens client-side, before anything is printed or sent.

## Step 2: submit

```bash
uv run skulk-harness submit runs/<your-run>
```

On success the command prints the ingest API's JSON response, including your
submission's id and status.

## What happens after you submit

1. **Validation gates.** The ingest checks the payload is a well-formed,
   attributable run (see the gate list below). A rejected submission is
   returned immediately with the reason; nothing is stored.
2. **Manual review queue.** Accepted submissions wait for a human approval
   pass. This is the ledger's honesty filter: it keeps spam and obviously
   broken runs off the public site.
3. **Publication.** Approved runs appear on
   [benchmarks.foxlight.ai](https://benchmarks.foxlight.ai), badged with your
   GitHub handle and the hardware class from your fingerprint.

Two server-side limits to know about:

- **Duplicates**: submitting the same run twice returns `409`. The run id is
  the dedup key (which is why the label redaction preserves it
  deterministically).
- **Quota**: submissions are rate-limited per submitter per hour. If you hit
  the quota, wait and resubmit; nothing is lost.

## Troubleshooting

### `submit failed: no GitHub token`

The harness could not find a token anywhere in the precedence chain. Either
log in with the `gh` CLI (`gh auth login`), or export one:

```bash
export GH_TOKEN=<your token>
```

A `401` response from the ingest means a token was found but the ledger could
not use it to identify you; check that the token is valid and not expired.

### `ingest rejected the submission (422): ...`

A `422` means the payload failed a validation gate. The response says which
one. The gates and what they mean:

| Gate | Meaning | Usual fix |
| --- | --- | --- |
| Run id shape | The `run_id` does not look like a harness run id (`YYYYMMDD-HHMMSS-<name>`) | Submit an unmodified `report.json` written by the harness |
| Results present | The report has no test results | You submitted a dry-run plan; rerun with `--execute` and submit that |
| Fingerprint with nodes | The report has no fingerprint, or its fingerprint lists no nodes | The run predates fingerprinting, or every cluster probe failed during the run; rerun on a current harness against a reachable cluster |
| Schema 2.x | The fingerprint's `schema_version` is not a 2.x version | The report was produced by an old harness version; update and rerun |

In short: the ledger only accepts real, executed, self-describing runs from a
current harness. If your run is recent and executed, it passes.

### `ingest rejected the submission (409): ...`

This run was already submitted. That is fine: it means the ledger has it.
There is nothing to do.

### `stability-suite reports are not accepted by the community ledger`

You pointed `submit` at a stability run. Only model-scoring runs (from `run
--execute`) are submittable.

## Related pages

- [Reports](../reference/reports.md): what the fingerprint contains and why
- [Compare runs](compare-runs.md): checking your own numbers before sharing them
