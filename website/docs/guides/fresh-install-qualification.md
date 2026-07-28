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

## Physical fleet lifecycle

The release gate treats the operator's real Apple and AMD hardware as one
freshly installed topology. For each explicitly eligible physical fleet, the
harness:

1. acquires and rereads the authoritative fleet lease;
2. opens an independent SSH recovery channel to every declared member;
3. requires those members to be the complete live topology and refuses to stop
   a fleet with active model instances or runners;
4. creates checksummed, mode-600 recovery archives for every member on both the
   target and controller;
5. stops the existing Skulk service on every member;
6. installs the same pinned candidate into an empty temporary `HOME` on every
   member and runs the literal `cd "$HOME/skulk" && uv run skulk` command on
   default ports with no sandbox, product flags, or `SKULK_*` overrides;
7. requires the exact declared topology to form and remain stable, every member
   to report the pinned commit, its expected local backend and dashboard
   contract, generated `skulk.yaml`, and Zenoh DATA;
8. drives the dashboard and direct API through the declared entrypoint while
   ordinary placement selects compatible Apple or AMD members, continuously
   requiring the same complete node-identity set;
9. inspects a served engine on whichever compatible member placement selected
   and proves its shipped concurrency and unified-KV settings;
10. stops every temporary runtime and proves every temporary `HOME` is gone;
11. restores and verifies every original service, checkout, config hash,
    process arguments, API identity, and the complete original topology; and
12. releases the lease only after restoration succeeds and verifies the
    intended release against an authoritative remote reread.

Legacy single-target diagnostic legs can still declare paired isolation
commands. They are not the physical release gate. A target used by
`physical_fleets` declares `whole_fleet_member: true`, supplies no isolation
wrapper, and is never run individually by the default complete-matrix command.

The lease renews at one third of its TTL. Every renewal is followed by an
authoritative reread. A renewal or restoration failure stops further testing,
makes one emergency extension, leaves the lease held, and writes a critical
recovery report.

The SSH tunnels belong to the recovery control plane and run in separate
process sessions. An operator interrupt stops product work immediately, then
the harness defers any further termination signal until all temporary homes are
removed, all services are restored, tunnel teardown, provider deletion, and
lease handling finish. The final report still records the interruption as a
blocking outcome.

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
