---
title: Configuration
---

The harness config is a YAML file. By default, commands look for
`skulk-harness.yaml` in the repository root. If that file is missing, the CLI
uses safe built-in defaults.

## Public Example

The public starter config is:

```text
skulk-harness.example.yaml
```

Copy it for local use:

```bash
cp skulk-harness.example.yaml skulk-harness.yaml
```

`skulk-harness.yaml` is ignored by git so your local cluster URL and node
settings stay private.

## Minimal Config

```yaml
api_base_url: http://localhost:52415
model_sets_path: configs/model_sets.yaml
test_sets_path: configs/test_sets.yaml
output_dir: runs
cluster_nodes: {}
```

## Top-Level Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `api_base_url` | `http://localhost:52415` | Skulk API root used by live commands |
| `request_timeout_s` | `30` | Timeout for ordinary API requests |
| `generation_timeout_s` | `1800` | Overall timeout for long generations |
| `stream_read_timeout_s` | `120` | Max wait for the next streaming byte |
| `placement_ready_timeout_s` | `1800` | Max wait for a placed instance to become ready |
| `placement_ready_total_timeout_s` | unset | Hard ceiling on one model's entire readiness wait across every replacement instance; unset derives `2 * placement_ready_timeout_s + placement_appearance_timeout_s`. Hitting it fails loudly with `unavailable_reason: churn` |
| `placement_appearance_timeout_s` | `300` | Max wait for a requested placement to appear in state |
| `store_download_timeout_s` | `14400` | Max wait for `models download --wait` |
| `store_delete_timeout_s` | `30` | Max wait for best-effort staged model eviction |
| `poll_interval_s` | `2` | Delay between repeated state checks |
| `preview_settle_attempts` | `8` | Retries for transient placement preview gaps |
| `output_dir` | `runs` | Where reports are written; also where `compare` resolves run selectors |
| `model_sets_path` | `configs/model_sets.yaml` | YAML file containing model sets |
| `test_sets_path` | `configs/test_sets.yaml` | YAML file containing test sets |
| `cluster_nodes` | `{}` | SSH control settings for stability suites |

## Cluster Nodes

`cluster_nodes` is only needed for destructive stability suites.

```yaml
cluster_nodes:
  node-a:
    ssh_host: node-a
    kill_command: pkill -f "skulk"
    relaunch_command: cd /opt/skulk && ./scripts/run-skulk.sh
```

| Field | Meaning |
| --- | --- |
| `ssh_host` | SSH hostname or alias |
| `kill_command` | Shell command used to stop Skulk on that node |
| `relaunch_command` | Shell command used to relaunch Skulk on that node |
| `repo_path` | Backward-compatible fallback used only when `relaunch_command` is absent |

:::warning
Do not put real private SSH hostnames or machine-specific paths into public
example files. Use placeholders in examples and real values in ignored local
configs.
:::

## Multiple Configs

You can pass a config explicitly:

```bash
uv run skulk-harness tests sets --config skulk-harness.example.yaml
uv run skulk-harness tests sets --config examples/foxlight/skulk-harness.yaml
```

Use this pattern when switching between public defaults, Foxlight production,
and private local experiments.
