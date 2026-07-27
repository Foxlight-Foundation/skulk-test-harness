# Foxlight production profile

This directory is Foxlight Foundation's configured-fleet regression profile: the
model sets, test sets, config, and battery scripts that drive the 5-node
"kite" cluster feeding the public
[Skulk benchmarks ledger](https://benchmarks.foxlight.ai). It is kept public
as a worked example of a serious multi-node configuration, not as a starting
template and not as the release E2E gate.

Executed Foxlight regression runs require every eligible node to advertise
Zenoh as its resolved DATA transport. Operators provide the real eligible
friendly-name inventory in an ignored local config and pass it with
`SKULK_HARNESS_CONFIG`; this keeps incidental fabric members out of previews,
reused instances, and final placement without publishing private inventory.
The battery refuses to exercise a path different from the one Skulk ships, but
its already-configured fleet still cannot substitute for
`fresh-install qualify`.

The MLX concurrency cells likewise stop at Skulk's shipped admission width of
8 when no product override is present. The `concurrency-16` and
`concurrency-reasoning-16` suites remain available for an explicit operator
policy, but their 16-way results are invalid unless every eligible MLX node is
actually launched with `SKULK_MAX_CONCURRENT_REQUESTS=16`.

The scripts assume that specific fleet (its node names, model store contents,
and hardware mix) and will not run elsewhere unmodified. For your own setup,
start from the
[quickstart](https://foxlight-foundation.github.io/skulk-test-harness/quickstart)
and `skulk-harness.example.yaml` at the repo root; if several operators share
one fleet, see the
[fleet coordination guide](https://foxlight-foundation.github.io/skulk-test-harness/guides/fleet-coordination).

The safe inventory shape for release qualification is documented in
`fresh-install.example.yaml`. Real SSH aliases, service commands,
keys, and the fleet-lock remote belong in an ignored local config.

Run the configured battery with that local inventory:

```bash
SKULK_HARNESS_CONFIG=/path/to/local-skulk-harness.yaml \
  bash examples/foxlight/run_e2e_battery.sh
```
