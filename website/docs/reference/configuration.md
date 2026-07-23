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
| `required_data_transport` | unset | Optional release-qualification gate: require every live node represented in either `/state` `nodeResources` or `nodeIdentities` telemetry to advertise `zenoh` or `gossipsub` before an executed run can mutate the cluster |
| `cluster_nodes` | `{}` | SSH control settings for stability suites |
| `fresh_install` | unset | Opt-in release qualification inventory and lifecycle policy |

## Required Data Transport

Generic and community profiles leave `required_data_transport` unset. A
release-qualification profile can pin the transport that Skulk ships:

```yaml
required_data_transport: zenoh
```

Before any named run, natural-language goal, or stability suite performs a
mutating action, the harness reads `/state` and checks every live node present
in either `nodeResources` or `nodeIdentities`. Execution stops before placement
or other cluster changes if a node has no transport advertisement, if the
telemetry maps are only partially populated, or if any node reports a different
transport.

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

## Fresh-install Inventory

Fresh qualification has a separate inventory from `cluster_nodes`. Selection
uses only entries with `eligible: true`; a fabric peer that is not in this map
is irrelevant.

```yaml
fresh_install:
  required_platforms: [apple, amd, nvidia]
  snapshot_root: fresh-install-snapshots
  snapshot_retention_days: 30
  lease_ttl_s: 3600
  emergency_lease_ttl_s: 21600
  targets:
    apple:
      kind: physical
      platform: apple
      hardware_class: apple-silicon-32gb
      eligible: false
      exclusion_reason: replace placeholders before enabling
      ssh_host: replace-me
      service_manager: launchd
      service_stop_command: replace-me
      service_start_command: replace-me
      isolation_enter_command: replace-with-target-local-skulk-traffic-isolation
      isolation_exit_command: replace-with-target-local-isolation-removal
      expected_backends: [mlx, mlx-metal]
      expected_data_transport: zenoh
      vision_contract: positive
      text_models:
        - mlx-community/Qwen3.5-2B-4bit
        - mlx-community/Qwen3-VL-4B-Instruct-4bit
      vision_models:
        - mlx-community/Qwen3.5-2B-4bit
        - mlx-community/Qwen3-VL-4B-Instruct-4bit
```

When `fresh-install qualify` is run without `--target`, every
`required_platforms` entry must have at least one explicitly eligible target or
the release matrix is refused before any lifecycle mutation begins. Repeated
`--target` options are for deliberate single-leg qualification and do not claim
the complete release status.

The heartbeat defaults to one third of `lease_ttl_s` and cannot be configured
less safely. Physical targets also declare config paths and an existing
checkout for hash/commit restoration checks. RunPod settings include its
neutral image, SSH keys, GPU choices, maximum hourly price and runtime, and
never include a network volume.

Eligible physical targets must also provide reversible isolation commands.
They block only Skulk discovery and fabric traffic on the selected target while
preserving SSH. This is how a default, override-free temporary process is
required to observe exactly one node even when the ordinary dev fleet remains
online. Isolation removal is part of mandatory restoration; failure leaves the
lease held.
