"""Git-backed fleet lease: a mutex over the shared Skulk test fleet.

Two agents work the Skulk codebase concurrently and both deploy branches to one
shared test fleet. Skulk does not support mixed-version clusters, so two
end-to-end runs on the same fleet at once corrupt each other silently. This
module is a mutex over the fleet, backed by a small JSON file
(``coordination/fleet-lock.json``) in a shared git repo.

The mutex *is* git. Acquiring means committing the claim and pushing; a push
rejected as non-fast-forward means another agent acquired first, so there is no
race: git's own atomic ref update is the compare-and-swap. A TTL
(``expires_at``) is the safety valve so a crashed run that never releases cannot
wedge the fleet forever, since a lock past its expiry is treated as free.

The lease is opt-in. With no ``fleet_lock`` config the store is never
constructed, so community users of the public harness are unaffected.
"""

from __future__ import annotations

import fcntl
import json
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from skulk_test_harness.models import FleetLock

_GIT_OPERATION_TIMEOUT_S = 30.0
_GIT_FETCH_RETRY_DELAYS_S = (1.0, 3.0)


def _utcnow() -> datetime:
    """Current UTC time (wall clock; the harness runs on a real machine)."""

    return datetime.now(UTC)


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp, treating a naive value as UTC.

    Returns ``None`` for missing or unparseable input so a malformed lock never
    crashes a caller; the caller then treats it as "no expiry known".
    """

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _default_cache_dir() -> Path:
    """Default clone location for the coordination repo."""

    return Path.home() / ".cache" / "skulk-test-harness" / "fleet-lock"


class FleetLease(BaseModel):
    """In-memory view of ``coordination/fleet-lock.json``.

    ``extra="ignore"`` keeps an older harness forward-compatible with a lock file
    that gains fields, so a newer coordinator does not break older readers.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # ``schema`` shadows a BaseModel attribute name, so store it as
    # ``schema_version`` with the on-disk alias ``schema``.
    schema_version: int = Field(default=1, alias="schema")
    state: str = "free"
    # The named fleet segment this lease covers ("main" when absent). Older
    # harnesses ignore the field via extra="ignore"; older lock files without
    # it read back as the main segment.
    segment: str | None = None
    holder: str | None = None
    branch: str | None = None
    host: str | None = None
    battery: str | None = None
    acquired_at: str | None = None
    expires_at: str | None = None
    heartbeat_at: str | None = None
    note: str | None = None

    def expiry(self) -> datetime | None:
        """The lease expiry as a datetime, or ``None`` if absent/unparseable."""

        return _parse_iso(self.expires_at)

    def is_expired(self, now: datetime) -> bool:
        """Whether a held lease has passed its TTL and is therefore reclaimable.

        A held lock with no parseable expiry is treated as expired: a lock we
        cannot reason about must not be able to wedge the fleet forever.
        """

        if self.state != "held":
            return False
        expiry = self.expiry()
        if expiry is None:
            return True
        return now >= expiry

    def is_held(self, now: datetime) -> bool:
        """Whether the fleet is actively held (held and not past its TTL)."""

        return self.state == "held" and not self.is_expired(now)

    def to_json(self) -> str:
        """Serialize to the on-disk JSON shape (aliased keys, trailing newline)."""

        return json.dumps(self.model_dump(by_alias=True), indent=2) + "\n"


@dataclass(frozen=True)
class LeaseOutcome:
    """Result of a lease operation.

    ``ok`` is whether the operation achieved its goal (acquired / extended /
    released). ``lease`` is the authoritative lease after the operation, and
    ``message`` is a human-readable explanation for the CLI.
    """

    ok: bool
    lease: FleetLease
    message: str


SEGMENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
"""Segment names stay short, lowercase, and path-safe."""


def segment_lock_path(base_path: str, segment: str) -> str:
    """Derive the lock-file path for a named segment.

    The main segment keeps the configured path unchanged, so existing
    deployments and older harness versions are untouched; any other segment
    inserts its name before the extension (``fleet-lock.den.json``), giving
    each segment an independent compare-and-swap file in the same repo.
    """

    if segment == MAIN_SEGMENT:
        return base_path
    if not SEGMENT_NAME_PATTERN.match(segment):
        raise ValueError(
            f"invalid segment name {segment!r}: use 1-32 chars of [a-z0-9-], "
            "starting alphanumeric"
        )
    if base_path.endswith(".json"):
        return f"{base_path[: -len('.json')]}.{segment}.json"
    return f"{base_path}.{segment}"


MAIN_SEGMENT = "main"
"""The default segment: the shared pool, stored at the configured path."""


class FleetLockStore:
    """Git compare-and-swap store for the fleet lease.

    Backed by a local clone of the coordination repo. Reads reset the clone to
    the remote branch first; writes commit and push, and a rejected push is
    surfaced as a lost race rather than retried blindly.
    """

    def __init__(self, config: FleetLock, *, segment: str = MAIN_SEGMENT) -> None:
        self._config = config
        self._segment = segment
        self._lock_path = segment_lock_path(config.path, segment)
        self._dir = (config.cache_dir or _default_cache_dir()).expanduser()
        self._local_lock_path = self._dir.parent / f".{self._dir.name}.lock"

    @contextmanager
    def _local_cache_lock(self) -> Iterator[None]:
        """Serialize every process that mutates or resets this local clone.

        Git provides the cross-host compare-and-swap at push time, but every
        process on one controller shares the same cache clone. A concurrent
        read runs ``reset --hard`` during synchronization; without this local
        lock it can erase another process's pending lease write before that
        writer commits it, causing a false-successful no-op release.
        """

        self._local_lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._local_lock_path.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    # --- git plumbing ---------------------------------------------------------

    def _git(
        self, *args: str, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self._dir,
            check=check,
            capture_output=True,
            text=True,
            timeout=_GIT_OPERATION_TIMEOUT_S,
        )

    def _ensure_clone(self) -> None:
        """Ensure a clean clone at the remote branch tip."""

        if (self._dir / ".git").is_dir():
            self._fetch_authoritative_branch()
            self._git(
                "reset", "--quiet", "--hard", f"origin/{self._config.branch}"
            )
            return
        self._dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth",
                "1",
                "--branch",
                self._config.branch,
                self._config.remote,
                str(self._dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_OPERATION_TIMEOUT_S,
        )

    def _fetch_authoritative_branch(self) -> None:
        """Fetch the lease branch with bounded transient transport retries.

        Fetch is read-only and therefore safe to repeat after a timeout or
        transport error. The lease operation still fails closed when every
        attempt fails, and callers still reset and reread the authoritative
        record before trusting it.
        """

        retry_delays = (*_GIT_FETCH_RETRY_DELAYS_S, None)
        for retry_delay in retry_delays:
            try:
                self._git("fetch", "--quiet", "origin", self._config.branch)
                return
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                if retry_delay is None:
                    raise
                time.sleep(retry_delay)

    def _commit_and_push(self, lease: FleetLease, message: str) -> bool:
        """Write, commit, and push the lease. Return False on non-fast-forward.

        A rejected push means another writer advanced the branch first (the
        compare-and-swap lost). The clone is reset back to the remote tip so the
        caller can re-read the authoritative state.
        """

        lock_path = self._dir / self._lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(lease.to_json())
        self._git("add", "--", self._lock_path)
        # Idempotent no-op writes (identical content) need no commit or push.
        if not self._git("status", "--porcelain").stdout.strip():
            # A no-op is successful only when an authoritative reread contains
            # the intended lease. This catches interference from an older
            # harness process that does not yet honor the local cache lock.
            return self._read_unlocked() == lease
        self._git(
            "-c",
            "user.name=skulk-harness",
            "-c",
            "user.email=harness@foxlight.local",
            "commit",
            "--quiet",
            "-m",
            message,
        )
        push = self._git(
            "push", "origin", f"HEAD:{self._config.branch}", check=False
        )
        if push.returncode == 0:
            authoritative = self._read_unlocked()
            if authoritative == lease:
                return True
            # A new holder can legitimately acquire immediately after a
            # successful release push. The release still achieved its goal.
            return (
                lease.state == "free"
                and authoritative.state == "held"
                and authoritative.holder != self._config.holder
            )
        # Lost the CAS: reset to the remote tip so a subsequent read is truthful.
        self._git("fetch", "--quiet", "origin", self._config.branch, check=False)
        self._git(
            "reset",
            "--quiet",
            "--hard",
            f"origin/{self._config.branch}",
            check=False,
        )
        return False

    # --- lease operations -----------------------------------------------------

    def _read_unlocked(self) -> FleetLease:
        """Fetch the lease while the caller owns the local cache lock."""

        self._ensure_clone()
        lock_path = self._dir / self._lock_path
        if not lock_path.exists():
            return FleetLease()
        return FleetLease.model_validate_json(lock_path.read_text())

    def read(self) -> FleetLease:
        """Fetch the latest lock and return the current lease (free if absent)."""

        with self._local_cache_lock():
            return self._read_unlocked()

    def acquire(
        self,
        *,
        branch: str,
        host: str,
        battery: str | None = None,
        ttl_s: float | None = None,
        note: str | None = None,
    ) -> LeaseOutcome:
        """Acquire the fleet, or refuse if another agent holds it unexpired."""

        with self._local_cache_lock():
            return self._acquire_locked(
                branch=branch,
                host=host,
                battery=battery,
                ttl_s=ttl_s,
                note=note,
            )

    def _acquire_locked(
        self,
        *,
        branch: str,
        host: str,
        battery: str | None,
        ttl_s: float | None,
        note: str | None,
    ) -> LeaseOutcome:
        """Acquire while holding the controller-local cache lock."""

        current = self._read_unlocked()
        now = _utcnow()
        if current.is_held(now) and current.holder != self._config.holder:
            return LeaseOutcome(
                False,
                current,
                f"fleet held by {current.holder} "
                f"(branch {current.branch}, expires {current.expires_at})",
            )
        took_over_stale = (
            current.state == "held"
            and current.holder != self._config.holder
            and current.is_expired(now)
        )
        ttl = ttl_s if ttl_s is not None else self._config.default_ttl_s
        lease = FleetLease(
            state="held",
            segment=self._segment,
            holder=self._config.holder,
            branch=branch,
            host=host,
            battery=battery,
            acquired_at=now.isoformat(),
            expires_at=(now + timedelta(seconds=ttl)).isoformat(),
            heartbeat_at=now.isoformat(),
            note=note,
        )
        if not self._commit_and_push(
            lease, f"fleet: {self._config.holder} acquire ({branch})"
        ):
            return LeaseOutcome(
                False,
                self._read_unlocked(),
                "lost the acquire race; another agent claimed the fleet first",
            )
        message = "acquired the fleet"
        if took_over_stale:
            message = "acquired the fleet (took over an expired stale lock)"
        return LeaseOutcome(True, lease, message)

    def extend(self, *, ttl_s: float | None = None) -> LeaseOutcome:
        """Push the TTL forward. Only the current holder may extend."""

        with self._local_cache_lock():
            return self._extend_locked(ttl_s=ttl_s)

    def _extend_locked(self, *, ttl_s: float | None) -> LeaseOutcome:
        """Extend while holding the controller-local cache lock."""

        current = self._read_unlocked()
        now = _utcnow()
        if current.state != "held" or current.holder != self._config.holder:
            return LeaseOutcome(
                False,
                current,
                "cannot extend: you do not currently hold the fleet",
            )
        ttl = ttl_s if ttl_s is not None else self._config.default_ttl_s
        lease = current.model_copy(
            update={
                "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
                "heartbeat_at": now.isoformat(),
            }
        )
        if not self._commit_and_push(
            lease, f"fleet: {self._config.holder} extend"
        ):
            return LeaseOutcome(
                False,
                self._read_unlocked(),
                "lost the extend race; re-read the lease",
            )
        return LeaseOutcome(True, lease, "extended the lease")

    def release(self, *, force: bool = False) -> LeaseOutcome:
        """Free the fleet. Only the holder may release without ``force``."""

        with self._local_cache_lock():
            return self._release_locked(force=force)

    def _release_locked(self, *, force: bool) -> LeaseOutcome:
        """Release while holding the controller-local cache lock."""

        current = self._read_unlocked()
        if current.state == "free":
            return LeaseOutcome(True, current, "fleet is already free")
        if not force and current.holder != self._config.holder:
            return LeaseOutcome(
                False,
                current,
                f"cannot release: fleet held by {current.holder} "
                "(use force to break it)",
            )
        broke_other = current.holder != self._config.holder
        free = FleetLease()
        if not self._commit_and_push(
            free, f"fleet: {self._config.holder} release"
        ):
            latest = self._read_unlocked()
            if latest.state == "free":
                return LeaseOutcome(True, latest, "fleet is free")
            return LeaseOutcome(
                False, latest, "lost the release race; re-read the lease"
            )
        message = "released the fleet"
        if force and broke_other:
            message = f"force-released the fleet (was held by {current.holder})"
        return LeaseOutcome(True, free, message)
