---
title: Run The Foxlight Profile
---

Foxlight's production harness profile lives under `examples/foxlight/`. It is a
real profile, not a placeholder: these batteries drive the fleet whose runs
feed the public [benchmarks ledger](https://benchmarks.foxlight.ai). It keeps
Foxlight-specific model matrices, test batteries, and benchmark notes out of
the public defaults while preserving existing automation, and it doubles as a
worked example of a serious configuration.

## The Important Files

| File | Purpose |
| --- | --- |
| `examples/foxlight/skulk-harness.yaml` | Real Foxlight e2e config |
| `examples/foxlight/model_sets.yaml` | Foxlight model sets and matrices |
| `examples/foxlight/test_sets.yaml` | Foxlight test batteries |
| `examples/foxlight/run_e2e_battery.sh` | Full e2e battery |
| `examples/foxlight/run_mtp_battery.sh` | Served speculation and MTP battery |
| `examples/foxlight/run_throughput_battery.sh` | Throughput battery |
| `examples/foxlight/skulk-harness.stability.example.yaml` | Optional destructive stability example |

## Root Scripts Still Work

The root scripts are compatibility wrappers:

```bash
./run_e2e_battery.sh
./run_mtp_battery.sh
./run_throughput_battery.sh
```

Each wrapper calls the matching script under `examples/foxlight/`, and the
Foxlight scripts pass:

```bash
--config examples/foxlight/skulk-harness.yaml
```

That means existing Foxlight automation can keep using the old root names.

## Direct Foxlight Commands

You can inspect the Foxlight model and test sets directly:

```bash
uv run skulk-harness models sets --config examples/foxlight/skulk-harness.yaml
uv run skulk-harness tests sets --config examples/foxlight/skulk-harness.yaml
```

You can also run one Foxlight set by name:

```bash
uv run skulk-harness run \
  --config examples/foxlight/skulk-harness.yaml \
  --model-set smoke \
  --test-set chat-tests \
  --execute \
  --delete-created-instances
```

Replace the model set and test set with the names printed by the list commands.

## Production Safety Checklist

Before running a production battery:

| Check | Why |
| --- | --- |
| Confirm the target API | Avoid running against the wrong cluster |
| Check current cluster work | Avoid interrupting active benchmarks |
| Use the wrapper scripts for standard batteries | Keep automation behavior consistent |
| Read the final report directory | Capture failures before rerunning |
| Avoid destructive stability suites unless scheduled | They can kill or relaunch Skulk processes |

## Stability Is Separate

The stability example is intentionally named:

```text
examples/foxlight/skulk-harness.stability.example.yaml
```

Copy it to a private file and fill in node-specific SSH settings before using
destructive stability suites.

Do not put real private machine paths or internal-only hostnames into a public
example file.
