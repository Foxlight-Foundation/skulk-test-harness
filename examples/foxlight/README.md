# Foxlight production profile

This directory is Foxlight Foundation's own production harness profile: the
model sets, test sets, config, and battery scripts that drive the 5-node
"kite" cluster feeding the public
[Skulk benchmarks ledger](https://benchmarks.foxlight.ai). It is kept public
as a worked example of a serious multi-node configuration, not as a starting
template.

Executed Foxlight runs require every live node to advertise Zenoh as its
resolved DATA transport. This is an intentional release-qualification gate:
the battery refuses to certify a fleet running a different transport from the
one a fresh Skulk installation uses by default.

The scripts assume that specific fleet (its node names, model store contents,
and hardware mix) and will not run elsewhere unmodified. For your own setup,
start from the
[quickstart](https://foxlight-foundation.github.io/skulk-test-harness/quickstart)
and `skulk-harness.example.yaml` at the repo root; if several operators share
one fleet, see the
[fleet coordination guide](https://foxlight-foundation.github.io/skulk-test-harness/guides/fleet-coordination).
