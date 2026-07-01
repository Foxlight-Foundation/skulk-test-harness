---
title: Stability Suites
---

Stability suites test operational behavior, not only model answers. They are for
operators who understand the cluster they are testing.

## Suites

| Suite | Command | What it does | Destructive |
| --- | --- | --- | --- |
| Soak | `stability soak` | Sends sustained concurrent completions | No SSH kill or relaunch |
| Failover | `stability failover` | Crashes the master mid-stream and checks recovery | Yes |
| Churn | `stability churn` | Repeatedly crashes and relaunches a non-master node | Yes |
| Refusal | `stability refusal` | Exercises impossible placement behavior | Yes |

## The Stability Battery

`run_stability_battery.sh` orchestrates these suites the way
`run_e2e_battery.sh` orchestrates the model matrix. Where the e2e battery proves
each model works *once*, the stability battery proves the cluster holds up under
*sustained* load and node loss.

```bash
./run_stability_battery.sh                # soaks only, non-destructive, safe
./run_stability_battery.sh --destructive  # + failover, churn, refusal
```

It runs three soak cells by default:

| Cell | Engine | Why it matters |
| --- | --- | --- |
| `mlx-baseline` | MLX (in-process) | Regression anchor: the mature engine under load |
| `amd-llama-cpp` | llama.cpp (in-process, GPU node) | Sustained GGUF decode on the AMD node |
| `amd-served-mtp` | llama-server `--spec-type draft-mtp` (GPU node) | The headline AMD path under load |

The **AMD cells are the point**. On an AMD Strix Halo deployment a single GPU
node carries the whole serving load a Mac cluster would spread across many, and
that audience has no MLX-style fallback. So the battery soaks the AMD llama.cpp
and served-MTP engines under concurrency and watches for a wedge, memory creep,
or an orphaned `llama-server` subprocess, not just a single happy-path request.
Each cell pre-stages its model into the store first, so the soak spends its
window on load rather than a first download.

The three soaks are non-destructive and run by default. The destructive trio
(failover, churn, refusal) runs only with `--destructive` and needs a stability
config whose `cluster_nodes` carry real `ssh_host` / `kill_command` /
`relaunch_command` values (see [Configure Nodes](#configure-nodes)); point the
`STABILITY_CONFIG` environment variable at it. Other knobs: `SOAK_CONCURRENCY`,
`MLX_SOAK_S`, `AMD_SOAK_S`, and `SKULK_STORE_URL`.

## Safety Flag

The destructive suites require an explicit opt-in:

```bash
uv run skulk-harness stability failover --execute-destructive
uv run skulk-harness stability churn --execute-destructive
uv run skulk-harness stability refusal --execute-destructive
```

Without `--execute-destructive`, the harness exits before any API or SSH side
effects for those suites.

:::warning
Do not run destructive stability commands on a shared or production cluster
unless the people using that cluster know it is about to happen.
:::

## Configure Nodes

Destructive suites need SSH control surfaces in `cluster_nodes`.

```yaml
cluster_nodes:
  kite1:
    ssh_host: kite1
    kill_command: pkill -f "skulk"
    relaunch_command: cd /opt/skulk && ./scripts/run-skulk.sh
  kite2:
    ssh_host: kite2
    kill_command: pkill -f "skulk"
    relaunch_command: cd /opt/skulk && ./scripts/run-skulk.sh
```

The map key should match the friendly node name reported by Skulk cluster
state. `ssh_host` is the host or alias passed to `ssh`.

## Backward-Compatible Relaunch

Older private configs may use `repo_path` instead of `relaunch_command`:

```yaml
cluster_nodes:
  local-node:
    ssh_host: local-node
    repo_path: /opt/skulk
```

That fallback still exists, but new configs should prefer explicit
`kill_command` and `relaunch_command`. Explicit commands are easier to review
and safer for open-source examples.

## Run A Non-Destructive Soak

Soak does not use the destructive opt-in:

```bash
uv run skulk-harness stability soak \
  --model mlx-community/Qwen3.5-9B-4bit \
  --concurrency 4 \
  --duration-s 120
```

Soak still sends real live requests. Use a model your cluster can serve and a
duration that is appropriate for the environment.

## Read Stability Reports

Stability reports write:

| File | Meaning |
| --- | --- |
| `report.json` | Structured suite result |
| `summary.md` | Human-readable observations, latency, and issues |

The report has a top-level `passed` value. Any error-severity issue marks the
suite as failed.
