---
title: Fresh-install release qualification
---

Fresh-install qualification is the only harness suite that can satisfy Skulk's
release E2E gate. The older battery connects to an already configured cluster;
it remains configured-fleet regression coverage for multi-node routing,
failover, concurrency, remote vision transport, and performance.

## Profiles

`candidate` installs a full expected commit SHA from `dev`. The installer
fetches that exact object and checks it out detached, so a moving branch cannot
change the tested candidate.

`shipping` runs the literal public README command against `main`, with no
product flags or `SKULK_*` environment overrides. The expected promoted commit
is supplied only as a post-install assertion, so a moving `main` cannot publish
status for different code. It runs after promotion but before a release or tag
is published.

```bash
uv run playwright install chromium

uv run skulk-harness fresh-install qualify \
  --profile candidate \
  --expected-commit <40-character-sha> \
  --config skulk-harness.fresh-install.yaml

uv run skulk-harness fresh-install qualify \
  --profile shipping \
  --expected-commit <40-character-promoted-main-sha> \
  --config skulk-harness.fresh-install.yaml
```

## Physical target lifecycle

For each explicitly eligible Apple or AMD target, the harness:

1. acquires and rereads the authoritative fleet lease;
2. creates checksummed, mode-600 recovery archives on the target and controller;
3. stops only that target's configured Skulk service;
4. applies the target's reversible, SSH-preserving Skulk-network isolation;
5. installs into an empty temporary `HOME` and runs the printed
   `cd "$HOME/skulk" && uv run skulk` command on default ports;
6. reaches the API through an SSH tunnel and requires one-node topology,
   generated `skulk.yaml`, the expected backend, the served dashboard build,
   and Zenoh DATA;
7. drives dashboard and direct-API model journeys;
8. stops and removes the temporary installation;
9. removes isolation, restores the original service, and verifies checkout status, config hashes,
   process arguments, API identity, and fleet membership; and
10. releases the lease only after restoration succeeds.

The lease renews at one third of its TTL. Every renewal is followed by an
authoritative reread. A renewal or restoration failure stops further testing,
makes one emergency extension, leaves the lease held, and writes a critical
recovery report.

## RunPod lifecycle

The NVIDIA leg creates an ephemeral pod from a neutral CUDA image, provisions
Node and SSH as infrastructure prerequisites, attaches no network volume, and
rejects a provider price above the configured ceiling. A local deadline bounds
cost. Deletion always runs in `finally` and is polled until the provider returns
not found.

## Acceptance matrix

| Platform | Models | Vision |
| --- | --- | --- |
| Apple Silicon | `mlx-community/Qwen3.5-2B-4bit`, `mlx-community/Qwen3-VL-4B-Instruct-4bit` | Both must identify exact generated fixtures through dashboard and API |
| AMD Linux | `unsloth/Llama-3.2-1B-Instruct-GGUF` | Text succeeds and the dashboard does not offer vision |
| RunPod NVIDIA | `unsloth/Llama-3.2-1B-Instruct-GGUF` | Text succeeds, CUDA backend is detected, and the dashboard does not offer vision |

Positive vision uses different generated PNGs for browser and API. Each
contains an unpredictable six-character code and randomized color/shape. No
answer appears in the prompt and no judge model is used. Browser qualification
also proves the thumbnail appears before submission, the sent user message
retains its attachment, and the captured request data URL decodes to the exact
fixture digest.

## Artifacts

Fresh reports are private operational records. They retain installer and
runtime logs, generated configuration, fixture PNGs, Playwright traces and
screenshots, lifecycle transitions, lease expiries, snapshot checksums, and
restoration status. Additive publishable provenance contains only public
hardware class, commit/digest/backend/transport facts, and environment variable
names—never secret values, private paths, node names, or image bytes.
