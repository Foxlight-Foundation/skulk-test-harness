---
title: Quickstart
---

This page takes you from "I have a Skulk node running" to "I have a report
that says my cluster works". It assumes no prior knowledge of the harness.

## What you need

| Requirement | Why |
| --- | --- |
| A running Skulk node | The harness tests a live cluster through its API (default `http://localhost:52415`); it cannot start Skulk for you |
| [`uv`](https://docs.astral.sh/uv/) | Installs dependencies and runs the CLI |
| Python 3.13 or newer | The harness package targets Python 3.13; `uv` will fetch it if needed |
| At least one model in Skulk's model store | The first run uses whatever model your cluster already has |

If you do not have Skulk running yet, set it up first with the
[Skulk documentation](https://github.com/Foxlight-Foundation/Skulk). Come back
when a node is up and its dashboard or API answers.

## 1. Clone and install

```bash
git clone https://github.com/Foxlight-Foundation/skulk-test-harness
cd skulk-test-harness
uv sync
```

`uv sync` installs the harness into a project-local environment. Every command
on this site is run as `uv run skulk-harness ...` from the repository root.

To confirm the install worked without touching any cluster:

```bash
uv run skulk-harness models sets --config skulk-harness.example.yaml
uv run skulk-harness tests sets --config skulk-harness.example.yaml
```

These only read YAML and print two tables. The first lists the built-in
**model sets**: named groups of models to test, such as `store-smoke` (the
first model in your Skulk model store). The second lists the built-in **test
sets**: named groups of prompts and pass/fail checks, such as `chat-tests`
(basic chat questions with expected answers). Every harness run is one model
set paired with one test set.

## 2. Check the harness can see your cluster

```bash
uv run skulk-harness doctor
```

`doctor` calls your Skulk API and prints a compact table:

| Field | What it means | Good sign |
| --- | --- | --- |
| API | The URL the harness is using | It is your cluster |
| API node | The id of the node answering | Present |
| Known models | Models in Skulk's catalog | Greater than zero |
| Cluster memory nodes | Nodes reporting memory telemetry | Matches your node count |
| Instances | Model instances currently placed | Any value is fine |
| Runner states | Runner processes known to the cluster | Any value is fine |
| State drift issues | Inconsistencies between instances and runners | Zero |

If `doctor` prints this table, you are ready for step 4.

### If doctor cannot connect

`doctor` fails when the API is not reachable at the configured URL. Without a
config file, the harness assumes `http://localhost:52415`. Check, in order:

1. Is Skulk actually running? (Its own logs or dashboard will tell you.)
2. Is it on this machine, or on another host on your network?
3. Is it listening on a different port?

If the API lives anywhere other than `localhost:52415`, you need a config
file. That is step 3, which you should do anyway.

## 3. Create your config file

Copy the example:

```bash
cp skulk-harness.example.yaml skulk-harness.yaml
```

`skulk-harness.yaml` is ignored by git, so your cluster address and any
private settings stay on your machine. The defaults look like:

```yaml
api_base_url: http://localhost:52415
model_sets_path: configs/model_sets.yaml
test_sets_path: configs/test_sets.yaml
output_dir: runs
cluster_nodes: {}
```

If your Skulk API runs on another host or port, change `api_base_url` and
nothing else, then run `doctor` again until it connects. The other fields are
explained in the [configuration reference](reference/configuration.md).

## 4. Your first dry run

`run` is a **dry run by default**: it reads cluster state and reports what it
WOULD do, but does not place models or send prompts.

```bash
uv run skulk-harness run --model-set store-smoke --test-set chat-tests
```

- `store-smoke` is a model set that selects the first model currently in your
  Skulk model store.
- `chat-tests` is a test set with small chat prompts and simple checks (for
  example: "answer with the capital of France" must contain "Paris").

The command ends with a summary table like:

```text
       Harness Run 20260709-141530-store-smoke-chat-tests
  Field            Value
  Models           1
  Placements       1
  Results passed   0
  Results failed   0
  Issues           0
  Report dir       runs/20260709-141530-store-smoke-chat-tests
```

Read it like this:

- **Models: 1** means the model set resolved to one real model id from your
  store. If it is 0, your store is empty; see
  [First local run](guides/first-local-run.md) for adding a model.
- **Placements: 1** means the harness found a way to place (or reuse) that
  model on your cluster.
- **Results passed/failed are 0** because nothing executed. A dry run plans;
  it does not prompt.
- **Report dir** is where the plan was written. Every run, even a dry one,
  writes a report directory under `runs/`.

## 5. Your first real run

When the dry run looks sane, add two flags:

```bash
uv run skulk-harness run \
  --model-set store-smoke \
  --test-set chat-tests \
  --execute \
  --delete-created-instances
```

- `--execute` is the switch from planning to doing: the harness will place the
  model (or reuse an existing instance), wait for it to become ready, send
  each test prompt, and score the streamed answers.
- `--delete-created-instances` cleans up after itself: any model instance the
  harness started gets torn down at the end, leaving your cluster as it found
  it. Instances that already existed are left alone.

The first execute run can take a few minutes if Skulk needs to load the model
into memory. The summary table now shows real pass/fail counts.

## 6. Read the report

The run wrote a directory like:

```text
runs/20260709-141530-store-smoke-chat-tests/
  report.json
  events.jsonl
  summary.md
  artifacts/
```

Open `summary.md` first. It is the human-readable result:

| Section | What it tells you |
| --- | --- |
| Header | Run id, model set, test set, mode, start/finish times |
| Models | Which model ids were selected and how |
| Placements | Which instance and nodes served each model, and whether they were reused |
| Results | One row per model, test, and repetition: pass/fail, time to first token, tokens per second |
| Issues | Anything that went wrong, with severity and evidence |

If everything passed: your cluster placed a model, served real requests,
streamed answers, and gave correct ones. That is the whole point.

If something failed, the `Issues` section usually has the most direct clue,
and [Troubleshooting](troubleshooting.md) covers the common cases.

`report.json` is the same run in machine-readable form, including a
**fingerprint** of exactly what produced the numbers (Skulk version, node
hardware, cache state). The [reports reference](reference/reports.md) explains
every field.

## The short version

```bash
git clone https://github.com/Foxlight-Foundation/skulk-test-harness
cd skulk-test-harness
uv sync
cp skulk-harness.example.yaml skulk-harness.yaml   # set api_base_url if not localhost
uv run skulk-harness doctor
uv run skulk-harness run --model-set store-smoke --test-set chat-tests
uv run skulk-harness run --model-set store-smoke --test-set chat-tests --execute --delete-created-instances
```

## Where to next

| You want to... | Read this |
| --- | --- |
| Understand what just happened in more depth | [First local run](guides/first-local-run.md) |
| Test your own list of models | [Write a model set](guides/write-model-set.md) |
| Write your own checks | [Write a test set](guides/write-test-set.md) |
| Benchmark and compare two runs | [Compare runs](guides/compare-runs.md) |
| Share your results with the community | [Submit to the ledger](guides/submit-to-the-ledger.md) |
| Stress-test the cluster itself | [Stability suites](guides/stability-suites.md) |
