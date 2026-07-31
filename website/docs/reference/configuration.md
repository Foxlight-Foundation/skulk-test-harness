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
    apple-1: &apple-member
      kind: physical
      platform: apple
      hardware_class: apple-silicon-32gb
      eligible: false
      exclusion_reason: replace placeholders before enabling
      whole_fleet_member: true
      ssh_host: replace-apple-1
      service_manager: launchd
      service_stop_command: replace-me
      service_start_command: replace-me
      original_checkout: replace-with-existing-checkout
      original_config_paths:
        - replace-with-existing-skulk-yaml
        - replace-with-existing-skulk-env
      expected_backends: [mlx, mlx-metal, mlx_audio, mlx_audio-metal]
      expected_data_transport: zenoh
      vision_contract: positive
      text_models:
        - mlx-community/Qwen3.5-2B-4bit
        - mlx-community/Qwen3-VL-4B-Instruct-4bit
      vision_models:
        - mlx-community/Qwen3.5-2B-4bit
        - mlx-community/Qwen3-VL-4B-Instruct-4bit
      dashboard_audio:
        speech_synthesis_model: mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit
        transcription_model: mlx-community/parakeet-tdt-0.6b-v3
    apple-2:
      <<: *apple-member
      ssh_host: replace-apple-2
    amd-1:
      kind: physical
      platform: amd
      hardware_class: amd-vulkan
      eligible: false
      exclusion_reason: replace placeholders before enabling
      whole_fleet_member: true
      ssh_host: replace-amd-1
      service_manager: systemd
      service_stop_command: replace-me
      service_start_command: replace-me
      original_checkout: replace-with-existing-checkout
      original_config_paths:
        - replace-with-existing-skulk-yaml
        - replace-with-existing-skulk-env
      expected_backends: [llama_server, llama_server-vulkan]
      expected_data_transport: zenoh
      vision_contract: unavailable
      text_models:
        - unsloth/Llama-3.2-1B-Instruct-GGUF
  physical_fleets:
    release-fleet:
      hardware_class: mixed-apple-amd
      eligible: false
      exclusion_reason: replace every placeholder before enabling
      member_targets: [apple-1, apple-2, amd-1]
      entrypoint_target: apple-1
      qualification_targets: [apple-1, amd-1]
```

Eligible `launchd` and `systemd` targets must include the service's
`skulk.env` in `original_config_paths`. Recovery temporarily disables the
startup wrapper's auto-update through that file, waits for the restored API,
then reapplies every archived config byte before verifying the original
checkout and service state.

AMD and NVIDIA release targets also declare the effective served-engine
contract. The qualification inspects the fresh child process, drives a bounded
ordinary-chat burst, requires Skulk to observe overlapping work, and verifies
that the runner survives:

```yaml
expected_backends: [llama_server, llama_server-vulkan]
served_engine_contract:
  backend: llama_server-vulkan
  parallel: 16
  kv_unified: true
  probe_concurrency: 4
```

When `fresh-install qualify` is run without selection flags, every
`required_platforms` entry must be represented by either an eligible physical
fleet member or an explicitly eligible standalone target such as RunPod. The
release matrix is refused before mutation otherwise. `--physical-fleet` selects
a complete physical topology deliberately; repeated `--target` options retain
the legacy diagnostic/single-provider behavior and do not claim a complete
physical release status.

The heartbeat defaults to one third of `lease_ttl_s` and cannot be configured
less safely. Physical targets also declare config paths and an existing
checkout for hash/commit restoration checks. RunPod settings include its
neutral image, SSH keys, GPU choices, maximum hourly price and runtime, and
never include a network volume.

`dashboard_audio` is optional and valid only on a dashboard-serving target that
expects the `mlx_audio` backend. Both model IDs are release contracts: the
harness discovers, downloads, launches, and exercises them through the shipped
dashboard rather than treating missing speech capability as a skip.

Targets referenced by `physical_fleets` set `whole_fleet_member: true` and
provide no isolation commands or runtime wrapper. The harness stops every
declared member, starts their temporary installations with normal networking,
and requires that exact topology to form. Legacy one-node diagnostic targets
still require paired reversible isolation commands. Mixing the two modes is
rejected by configuration validation.
