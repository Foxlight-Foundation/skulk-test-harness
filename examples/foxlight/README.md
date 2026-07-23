# Foxlight production profile

This directory is Foxlight Foundation's configured-fleet regression profile: the
model sets, test sets, config, and battery scripts that drive the 5-node
"kite" cluster feeding the public
[Skulk benchmarks ledger](https://benchmarks.foxlight.ai). It is kept public
as a worked example of a serious multi-node configuration, not as a starting
template and not as the release E2E gate.

Executed Foxlight regression runs require every live node to advertise Zenoh
as its resolved DATA transport. The battery refuses to exercise a path
different from the one Skulk ships, but its already-configured fleet still
cannot substitute for `fresh-install qualify`.

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
