# **skulk-test-harness**

<div align="center">

[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FFoxlight-Foundation%2Fskulk-test-harness%2Fmain%2Fpyproject.toml&query=%24.project.version&prefix=v&label=version&color=blue&style=flat-square)](https://github.com/Foxlight-Foundation/skulk-test-harness/releases)
[![Tests](https://img.shields.io/github/actions/workflow/status/Foxlight-Foundation/skulk-test-harness/ci.yml?branch=main&label=tests&style=flat-square&logo=github)](https://github.com/Foxlight-Foundation/skulk-test-harness/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-4c72b0?style=flat-square)](LICENSE)

[![Documentation](https://img.shields.io/badge/docs-documentation-2ea44f?style=flat-square&logo=readthedocs&logoColor=white)](https://foxlight-foundation.github.io/skulk-test-harness/)
[![Quickstart](https://img.shields.io/badge/docs-quickstart-2ea44f?style=flat-square&logo=readthedocs&logoColor=white)](https://foxlight-foundation.github.io/skulk-test-harness/quickstart)
[![CLI Reference](https://img.shields.io/badge/docs-CLI_reference-2ea44f?style=flat-square&logo=readthedocs&logoColor=white)](https://foxlight-foundation.github.io/skulk-test-harness/reference/cli)

</div>

---

This harness has two deliberately separate jobs:

- **Fresh-install release qualification** installs Skulk from scratch and
  proves what a new user gets through both the dashboard and API.
- **Configured-fleet regression coverage** attaches to an already running
  cluster for routing, failover, concurrency, and benchmark work.

Only the first can satisfy Skulk's release E2E gate.

Point the harness at your cluster's API and it will place models, run real
chat/tool/vision/speech requests against them, measure time-to-first-token
and decode throughput, check the answers, and write an honest report you can
keep, compare against later runs, and (if you want) publish to the public
[Skulk benchmarks ledger](https://benchmarks.foxlight.ai).

## What you can do with it

- **Smoke-test a cluster**: "every model I care about serves a correct answer."
- **Benchmark**: wall-clock TTFT and tokens/second per model, per run, with
  the noise called out instead of hidden.
- **Compare runs**: like-for-like deltas between two runs, with trust guards
  that warn when a comparison is not actually fair.
- **Stress it**: soak, failover, churn, and refusal suites for operators who
  want to know what breaks first.
- **Share results**: one command submits a run to the community benchmarks
  ledger, redacted on your machine before anything leaves it.

## Five-minute start

You need: a running Skulk node (its API defaults to `http://localhost:52415`)
and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Foxlight-Foundation/skulk-test-harness
cd skulk-test-harness
uv sync
```

First, confirm the harness can see your cluster:

```bash
uv run skulk-harness doctor
```

`doctor` prints a compact summary of your nodes and models. If it cannot
reach the API, copy `skulk-harness.example.yaml` to `skulk-harness.yaml` and
set `api_base_url` to wherever your Skulk API lives.

Now preview a run. Nothing touches the cluster yet: `run` is a **dry run by
default**.

```bash
uv run skulk-harness run --model-set store-smoke --test-set chat-tests
```

That prints what WOULD happen: which models resolve, where they would be
placed, which tests would execute. When it looks right, let it actually run:

```bash
uv run skulk-harness run \
  --model-set store-smoke \
  --test-set chat-tests \
  --execute \
  --delete-created-instances
```

`--delete-created-instances` cleans up after itself: any model instance the
harness started gets torn down at the end, leaving your cluster as it found
it.

Release-qualification profiles can also set
`required_data_transport: zenoh` (or `gossipsub`). Before any named run,
natural-language goal, or stability suite mutates the cluster, the harness
checks every live node present in either `/state` telemetry map and refuses a
missing, mixed, or mismatched transport advertisement. Generic profiles leave
this unset.

## Fresh-install release qualification

Install Chromium once on the controller:

```bash
uv run playwright install chromium
```

Then run either the exact proposed `dev` commit or the literal public `main`
installer:

```bash
uv run skulk-harness fresh-install qualify \
  --profile candidate \
  --expected-commit <40-character-dev-commit> \
  --config skulk-harness.fresh-install.yaml

uv run skulk-harness fresh-install qualify \
  --profile shipping \
  --config skulk-harness.fresh-install.yaml
```

The inventory is opt-in: a target is ignored unless its local configuration
sets `eligible: true`. Physical targets are protected by the authoritative
fleet lease, dual recovery snapshots, verified restoration, and a lease
heartbeat. RunPod is created without a network volume and is deleted in
`finally`, with provider deletion polled to completion. See the
[fresh-install guide](https://foxlight-foundation.github.io/skulk-test-harness/guides/fresh-install-qualification).

## Where the results go

Every run writes a directory under `runs/`:

- `report.json`: the machine-readable record of every request, every metric,
  pass/fail, plus a **fingerprint** of exactly what ran it (Skulk version,
  node hardware, cache state), so a number is never separated from its
  context. It also carries the test set's description and each result's kind
  and description, so a downstream reader (the results ledger) can explain what
  a suite measures without the harness config.
- `summary.md`: the same story for humans.
- `events.jsonl` and `artifacts/`: the raw trail (speech tests keep their
  generated audio here).

Compare any two runs later:

```bash
uv run skulk-harness compare -b runs/<baseline> -n runs/<candidate>
```

`compare` shows per-model throughput deltas and refuses to pretend: if the
runs used different node sets, cache states, or too few samples, it says so.

## Share your results

The public [benchmarks ledger](https://benchmarks.foxlight.ai) collects runs
from the community, labeled by submitter and by the hardware that produced
them. Submitting is one command:

```bash
uv run skulk-harness submit runs/<your-run> --dry-run   # inspect the payload
uv run skulk-harness submit runs/<your-run>             # send it
```

Redaction happens **on your machine, before anything is sent**: generated
text, operator notes, run names, repo paths, API URLs, and node names never
leave it. `--dry-run` prints the exact payload so you can verify that
yourself. Submissions authenticate with your GitHub account (via the `gh`
CLI or a `GH_TOKEN`) and wait for manual approval before appearing on the
site.

## Model sets and test sets

Runs are named combinations of a **model set** (which models) and a **test
set** (which checks). List what is available:

```bash
uv run skulk-harness models sets
uv run skulk-harness tests sets
```

The built-in sets in `configs/` cover chat, code, tool calling, embeddings,
vision, speech (TTS/STT, streaming, realtime WebSocket, roundtrip), throughput,
cancellation, context admission, and served speculative decoding. Defining your
own is a few lines of YAML: see
[writing a model set](https://foxlight-foundation.github.io/skulk-test-harness/guides/write-model-set)
and
[writing a test set](https://foxlight-foundation.github.io/skulk-test-harness/guides/write-test-set).

## Safety defaults

- `run` and `goal` are dry runs unless you pass `--execute`.
- Fresh-install qualification is intentionally destructive to only the
  explicitly eligible target: it temporarily stops that target's existing
  Skulk service and refuses to release the lease until restoration is proved.
- The stability suites (`failover`, `churn`, `refusal`) additionally require
  `--execute-destructive` plus explicit SSH process-control configuration
  before they will touch anything. Soaks are non-destructive.
- Listing sets and configs never needs a live cluster; the offline test suite
  (`uv run pytest`) never touches one either.

## Coordinating a shared fleet

When more than one operator (or agent) deploys branches to the same test fleet,
two end-to-end runs at once collide: Skulk does not support mixed-version
clusters, so one deploy silently corrupts the other's run. The optional
**fleet lease** is a mutex over the fleet, backed by a small JSON file in a
shared git repo. It is off by default, so single-operator use is unaffected.

Enable it by adding a `fleet_lock` section to your config with the git remote
that holds the lock and a stable name for this operator:

```yaml
fleet_lock:
  remote: git@github.com:your-org/your-coordination-repo.git
  holder: operator-a           # your stable name; the other side uses another
  branch: main                 # optional (default: main)
  path: coordination/fleet-lock.json   # optional
  default_ttl_s: 1800          # optional; a lock past its TTL is treated as free
```

Bracket a fleet session with the lease:

```bash
uv run skulk-harness fleet acquire --branch feature/my-work
# ... deploy your branch to the fleet and run batteries ...
uv run skulk-harness fleet extend    # push the TTL forward on a long run
uv run skulk-harness fleet release
uv run skulk-harness fleet status    # see who holds it
```

The mutex is git itself: acquiring commits your claim and pushes, and a rejected
non-fast-forward push means the other side got it first (no race). The TTL is a
safety valve so a crashed run cannot wedge the fleet forever. As a backstop,
`run`/`goal`/stability commands refuse (in `--execute` mode) when another holder
holds the lease; pass `--force` to override. The
[fleet coordination guide](https://foxlight-foundation.github.io/skulk-test-harness/guides/fleet-coordination)
walks through the whole acquire/deploy/run/release bracket.

## Learn more

| | |
| --- | --- |
| [Quickstart](https://foxlight-foundation.github.io/skulk-test-harness/quickstart) | The five-minute start, with more hand-holding |
| [Concepts](https://foxlight-foundation.github.io/skulk-test-harness/concepts/harness-model) | How runs, sets, placements, and reports fit together |
| [Guides](https://foxlight-foundation.github.io/skulk-test-harness/guides/first-local-run) | First local run, custom sets, stability suites, submitting to the ledger |
| [Fresh-install qualification](https://foxlight-foundation.github.io/skulk-test-harness/guides/fresh-install-qualification) | The candidate and shipping release gates |
| [Fleet coordination](https://foxlight-foundation.github.io/skulk-test-harness/guides/fleet-coordination) | Sharing one test fleet across operators with the git-backed lease |
| [CLI reference](https://foxlight-foundation.github.io/skulk-test-harness/reference/cli) | Every command and flag |
| [Troubleshooting](https://foxlight-foundation.github.io/skulk-test-harness/troubleshooting) | When something looks wrong |

The site sources live under `website/` (Docusaurus); PRs build them, pushes
publish them.

## The Foxlight profile

Foxlight's attached configured-fleet regression matrix lives under
`examples/foxlight/`: the
`run_e2e_battery.sh`, `run_mtp_battery.sh`, `run_throughput_battery.sh`, and
`run_stability_battery.sh` entrypoints drive the fleet that feeds the public
ledger. They remain valuable, but they do not qualify a release or claim that a
fresh installation works. They are ordinary harness invocations and double as
worked examples of a serious configured fleet. See
[stability suites](https://foxlight-foundation.github.io/skulk-test-harness/guides/stability-suites)
for what the destructive ones do before running them anywhere.

## License

MIT
