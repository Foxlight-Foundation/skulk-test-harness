---
title: Coordinate A Shared Fleet
---

This guide is for organizations (or multi-agent setups) where more than one
operator deploys branches to the **same** physical test fleet. If you are the
only one touching your cluster, you do not need any of this: the fleet lease
is optional and disabled by default, and every `fleet` command is a no-op
until you configure it.

## The problem it solves

Skulk does not support mixed-version clusters: every node must run the same
build before serving workloads. So when two people (or two coding agents)
each deploy their own branch to one shared fleet and run end-to-end batteries
at the same time, one deploy silently corrupts the other's run. The results
look plausible and are wrong.

The **fleet lease** is a mutex over the fleet, backed by a small JSON file
(`coordination/fleet-lock.json` by default) in a git repository both sides
can push to. The mutex is git itself: acquiring the lease means committing a
claim and pushing it, and a push rejected as non-fast-forward means the other
side got there first. Git's atomic ref update is the compare-and-swap, so
there is no race window.

## Configuration

Add a `fleet_lock` section to your harness config (`skulk-harness.yaml`).
Both operators point at the same coordination repository; each uses their own
`holder` name:

```yaml
fleet_lock:
  # Git remote both operators can push to. Any small shared repo works;
  # a private one is typical.
  remote: https://github.com/your-org/your-coordination-repo.git

  # This operator's stable name. The other operator uses a different one.
  # Only the holder can extend or release the lease without --force.
  holder: operator-a

  # Optional. Branch carrying the lock file (default: main).
  branch: main

  # Optional. Path of the lock JSON inside the repo
  # (default: coordination/fleet-lock.json).
  path: coordination/fleet-lock.json

  # Optional. Default lease lifetime in seconds (default: 1800).
  # A lease past its TTL is treated as free.
  default_ttl_s: 1800
```

The harness keeps a local clone of the coordination repo (by default under
`~/.cache/skulk-test-harness/fleet-lock`; override with `cache_dir`). You
never touch that clone directly.

## The session bracket

Wrap every fleet session, from deploy to final teardown, in an
acquire/release bracket:

```bash
# 1. Take the fleet before deploying anything.
uv run skulk-harness fleet acquire --branch feature/my-work --battery e2e

# 2. Deploy your branch to the fleet and run your batteries.
uv run skulk-harness run --model-set store-smoke --test-set chat-tests --execute

# 3. On a long battery, push the TTL forward so the lease stays live.
uv run skulk-harness fleet extend --ttl-minutes 90

# 4. Free the fleet when you are done.
uv run skulk-harness fleet release

# At any point, see who holds it:
uv run skulk-harness fleet status
```

`fleet acquire` refuses (exit code 1) when the other operator holds an
unexpired lease, printing the holder, their branch, and the expiry. That
refusal is the feature: wait, or coordinate out-of-band.

As a backstop, the execute paths (`run --execute`, `goal --execute`, and the
stability suites) also refuse to touch the fleet while another holder has the
lease. The explicit bracket is still the primary mechanism, because it covers
the deploy itself, not just the harness runs. `--force` overrides the
backstop when you know the lease is wrong.

## TTL: the safety valve

Every lease has an expiry (`default_ttl_s`, or `--ttl-minutes` on
`acquire`/`extend`). A lease past its expiry is treated as **free**: a
crashed run that never released cannot wedge the fleet forever. Acquiring
over an expired lease is normal and reported as taking over a stale lock.

Two practical consequences:

- Long batteries must call `fleet extend` before the TTL runs out, or the
  other side may legitimately take the fleet mid-run.
- A malformed lock file (unparseable expiry) is also treated as expired, on
  the principle that a lock nobody can reason about must not hold the fleet.

## Troubleshooting a stale holder

**`fleet status` shows HELD but you believe nobody is running anything.**
First check `expires_at`: if it is in the past, the lease is already treated
as free (status says so) and a plain `fleet acquire` will take it over.

**The lease is unexpired but its holder is genuinely gone** (machine died,
run abandoned): confirm out-of-band that no deploy or battery is actually in
flight, then break the lock:

```bash
uv run skulk-harness fleet release --force
```

`--force` releases a lease held by someone else. Use it only after human
confirmation; the whole point of the lease is that the harness cannot tell a
slow battery from a dead one until the TTL expires.

**`acquire` reports a lost race.** The other operator pushed their claim
between your read and your push. Nothing is broken: re-run `fleet status`
and wait your turn.

## Related pages

- [Configuration](../reference/configuration.md): the rest of the harness config
- [Stability suites](stability-suites.md): the destructive suites that must
  never share a fleet with anything else
