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
checks every eligible node and refuses a missing, mixed, or mismatched
transport advertisement. Set `eligible_fleet_nodes` to stable friendly names
when the fabric can contain unrelated nodes; every listed node must be present,
and placement previews, reused instances, and final live placement are confined
to that allowlist. Incidental members are ignored. With no allowlist the
transport check retains its generic behavior and covers every live node.
Generic profiles leave both settings unset.

```yaml
required_data_transport: zenoh
eligible_fleet_nodes:
  - node-a
  - node-b
```

## Fresh-install release qualification

Install the release-gate browser engines once on the controller:

```bash
uv run playwright install chromium webkit
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
  --expected-commit <40-character-promoted-main-commit> \
  --config skulk-harness.fresh-install.yaml
```

If a run restores cleanly after every physical E2E cell passed but a
harness-only post-battery provenance gate failed, the corrected harness may
resume that one failed gate. A restored run whose only failure is the dashboard
release-experience topology count may also resume after the product fix:

```bash
uv run skulk-harness fresh-install qualify \
  --profile candidate \
  --expected-commit <same-40-character-dev-commit> \
  --resume-from <predecessor-fresh-install-report.json> \
  --config skulk-harness.fresh-install.yaml
```

For the dashboard boundary, `--expected-commit` is the new exact candidate.
The replacement run fresh-installs every physical member, proves the complete
topology again, reruns the primary model needed by the dashboard, and reruns
the failed dashboard cell. Green non-primary model cells are sealed from the
predecessor with a checksummed manifest; the complete E2E battery and RunPod
leg then run normally against the new candidate.

Post-battery resumption is fail-closed: the predecessor must prove the same candidate,
matrices, complete cell sequence, ordered physical platform/hardware/backend
contract, topology, all-green results, and a clean recorded harness source
tree. The battery script bytes resolved from that tree
must exactly match the current script, including cell commands and flags. The
normalized execution configuration must also match; only the tunneled API and
run-local artifact/matrix paths are excluded after matrix bytes are verified.
The corrected harness checkout must also remain clean and unchanged from
preflight through the resumed gate. The resumed run
still performs a normal whole-fleet fresh install and acceptance journey, then
seals and rechecks the predecessor cell evidence before continuing to the
mandatory clean RunPod leg. It cannot skip a product failure or combine
unrelated qualification legs.

The dashboard-cell boundary is separately fail-closed: it accepts only a
single topology-count failure after all model journeys passed, requires clean
restoration and the unchanged fleet/model contract, and refuses to resume the
same product commit. It does not convert missing nodes into an adaptive skip.

`fresh-install qualify` is atomic and always runs the complete eligible release
matrix. It holds one authoritative lease while it fresh-installs the physical
topology, runs the full E2E battery before restoring that topology, and then
provisions and deletes the mandatory RunPod/NVIDIA target. It emits one release
verdict and refuses partial selectors. Every E2E cell must prove fresh-install
provenance for the exact expected commit. The command validates the battery
script and both matrix files before acquiring the lease, then records private
mode-600 snapshots of the exact matrices consumed by the run.
To keep one cell's downloaded artifacts from consuming another cell's disk
budget, qualification evicts every harness-created model after its instance is
verified and torn down. Configured-fleet runs retain their warm-cache behavior
unless the operator explicitly requests the same cleanup.

Use `fresh-install diagnose --physical-fleet <name>` or
`fresh-install diagnose --target <name>` to debug one leg. A diagnostic pass is
never a release qualification and cannot satisfy the release gate.

The inventory is opt-in: a physical fleet or target is ignored unless its local
configuration sets `eligible: true`. The physical release gate stops every
declared member, installs all of them into empty temporary homes with normal
networking, and qualifies the topology they actually form—without a sandbox.
Every member is protected by the authoritative fleet lease, dual recovery
snapshots, verified all-node restoration, and a lease heartbeat. RunPod is a
mandatory part of the complete matrix, is created without a network volume,
and is deleted in `finally`, with provider deletion polled to completion. See the
[fresh-install guide](https://foxlight-foundation.github.io/skulk-test-harness/guides/fresh-install-qualification).
The browser gate also covers Settings save, topology rendering, persisted
conversations and attachments, a failed-request recovery, WebKit text chat,
and any explicitly configured dashboard TTS/STT contract.

After an automated candidate matrix passes, human acceptance exercises that
same exact commit as a first-time user. It supplements the automated verdict;
it cannot replace a failed physical-fleet cell, clean RunPod leg, restoration,
or provenance check. Follow Skulk's public
[human release qualification guide](https://github.com/Foxlight-Foundation/Skulk/blob/de74deb9b5cbb6cc31e4d91aaa71da5514d93192/website/docs/human-release-qualification.md).
If human testing leads to a product, installer, shipped-default, dashboard, or
model-card fix, the resulting commit is a new candidate and must repeat the
automated fresh-install matrix before promotion.

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
