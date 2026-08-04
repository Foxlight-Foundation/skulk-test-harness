---
title: Fresh-install release qualification
---

Fresh-install qualification is the only harness suite that can satisfy Skulk's
release E2E gate. The older battery connects to an already configured cluster;
it remains configured-fleet regression coverage for multi-node routing,
failover, concurrency, remote vision transport, and performance.

The release command is one atomic matrix. It holds one lease across the fresh
physical fleet, the complete E2E battery executed before that fleet is torn
down, and the clean RunPod/NVIDIA leg. It emits one composite verdict only after
every mandatory platform passes against the same exact commit. Individual legs
are diagnostics and cannot be combined later into a qualification.

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
uv run playwright install chromium webkit

uv run skulk-harness fresh-install qualify \
  --profile candidate \
  --expected-commit <40-character-sha> \
  --config skulk-harness.fresh-install.yaml

uv run skulk-harness fresh-install qualify \
  --profile shipping \
  --expected-commit <40-character-promoted-main-sha> \
  --config skulk-harness.fresh-install.yaml
```

## Narrow post-battery resumption

When every physical E2E cell passed and the fleet restored cleanly, but the
harness itself failed only while applying the final result/provenance gate, a
corrected harness can resume that exact failed stage:

```bash
uv run skulk-harness fresh-install qualify \
  --profile candidate \
  --expected-commit <same-40-character-sha> \
  --resume-from <predecessor-fresh-install-report.json> \
  --config skulk-harness.fresh-install.yaml
```

This is not a general skip-cells option. Before any fleet mutation, the harness
requires the predecessor to have exactly one failed lifecycle stage, the same
candidate commit, successful teardown and restoration, identical matrix bytes
and cell sequence, the same ordered anonymous platform/hardware/backend/
transport contract, one stable complete topology, and all-green result and
fresh-install provenance checks. The predecessor report must prove that its
harness checkout was clean; the immutable recorded commit/tree is resolved even
if that checkout has since advanced to the fix. Legacy reports that could not
distinguish clean from unknown require a still-clean checkout at the recorded
commit. The corrected checkout must remain clean at the same commit/tree from
preflight through the resumed gate. The harness seals checksummed copies of the
reports, requires the predecessor-commit battery script bytes to match the
current script exactly, and records both source identities in the new report.
The predecessor's generated battery configuration is normalized only for the
tunneled API endpoint and run-local output, lifecycle-report, and already
content-verified matrix paths. Every execution-affecting setting must match the
current configuration, and its normalized digest is sealed into the manifest.

The resumed qualification still begins with a normal whole-fleet fresh install
and repeats its installer, topology, backend, dashboard, API, vision, audio,
and served-engine acceptance. It advances only the already-completed E2E cells
to the failed provenance gate and then runs the mandatory clean RunPod/NVIDIA
leg. A product failure, changed candidate, changed matrix, incomplete battery,
failed recovery, or failure before the final provenance gate requires a full
new battery.

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
   member, created on that user's real home filesystem so Linux tmpfs-backed
   `/tmp` cannot distort model-storage capacity, and runs the literal
   `cd "$HOME/skulk" && uv run skulk` command on default ports with no sandbox,
   product flags, or `SKULK_*` overrides;
7. requires the exact declared topology to form and remain stable, every member
   to report the pinned commit, its expected local backend and dashboard
   contract, generated `skulk.yaml`, and Zenoh DATA;
8. drives the dashboard and direct API through the declared entrypoint while
   ordinary placement selects compatible Apple or AMD members, continuously
   requiring the same complete node-identity set;
9. opens and saves the installer-generated Settings, checks that every node is
   rendered in topology, reloads persisted text/image conversations, injects
   one failed chat request and proves a retry succeeds, and repeats a text
   smoke test with Playwright WebKit;
10. on a target declaring `dashboard_audio`, finds, downloads, and launches its
    TTS and STT models through the dashboard, requires real audio bytes from
    **Speak draft** plus a non-silent PCM duration, retains those exact bytes,
    then feeds the harness's known-speech WAV through Chromium's fake
    microphone and requires the transcript in the chat composer. TTS and STT
    use separate fixtures so a failure names the broken user journey instead
    of making one model's output the other model's input;
11. inspects a served engine on whichever compatible member placement selected
   and proves its shipped concurrency and unified-KV settings;
12. validates the battery script and matrix inputs before acquiring the lease,
    snapshots the exact model and test matrices as private artifacts, runs every
    complete E2E battery cell from those snapshots, and requires every child
    report to prove fresh-install provenance for the exact expected commit;
13. stops every temporary runtime and proves every temporary `HOME` is gone;
14. restores and verifies every original service, checkout, config hash,
    process arguments, and API identity, waiting until every member reports the
    same complete original topology before accepting recovery; and
15. keeps the lease through mandatory RunPod qualification and releases it only
    after restoration, provider deletion, and the composite audit succeed.

Legacy single-target diagnostic legs can still declare paired isolation
commands. They are not the physical release gate. A target used by
`physical_fleets` declares `whole_fleet_member: true`, supplies no isolation
wrapper, and is never run individually by the default complete-matrix command.
Selected debugging uses `fresh-install diagnose`; diagnostic reports never
produce a composite release verdict.

The lease renews at one third of its TTL. Every renewal is followed by an
authoritative reread. A renewal or restoration failure stops further testing,
makes one emergency extension, leaves the lease held, and writes a critical
recovery report.

Large batteries may set `physical_fleets.<name>.e2e_entrypoint_target` to a
different member of the same freshly installed topology. Dashboard, vision,
and audio user journeys continue through `entrypoint_target`; only the complete
direct-API E2E battery and its model-store downloads use the alternate fresh
member. This is useful when the release matrix is larger than the dashboard
member's local disk. The alternate target must be a declared fleet member and
does not relax commit, topology, backend, transport, or provenance checks.

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
not found. Omitting or failing this leg fails the composite qualification.

## Acceptance matrix

| Platform | Models | Vision and audio |
| --- | --- | --- |
| Apple Silicon | `mlx-community/Qwen3.5-2B-4bit`, `mlx-community/Qwen3-VL-4B-Instruct-4bit`, configured dashboard TTS/STT pair | Both chat models must identify exact generated fixtures through dashboard and API; dashboard TTS must return non-silent PCM audio and fake-microphone STT must recover the known fixture phrase |
| AMD Linux | `unsloth/Llama-3.2-1B-Instruct-GGUF` | Text succeeds and the dashboard does not offer vision |
| RunPod NVIDIA | `unsloth/Llama-3.2-1B-Instruct-GGUF` | Text succeeds, CUDA backend is detected, and the dashboard does not offer vision |

Positive vision uses different generated PNGs for browser and API. Each
contains an unpredictable six-character code and randomized color/shape. No
answer appears in the prompt and no judge model is used. Browser qualification
also proves the thumbnail appears before submission, the sent user message
retains its attachment, and the captured request data URL decodes to the exact
fixture digest. Reloading the page must retain the active user/assistant turn
and image attachment.

## Artifacts

Fresh reports are private operational records. They retain installer and
runtime logs, generated configuration, fixture PNGs, Playwright traces and
screenshots, lifecycle transitions, lease expiries, snapshot checksums, and
restoration status. Additive publishable provenance contains only public
hardware class, commit/digest/backend/transport facts, and environment variable
names—never secret values, private paths, node names, or image bytes.
