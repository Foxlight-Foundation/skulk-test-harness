"""Authoritative fleet-lease verification and renewal for long qualifications."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from skulk_test_harness.fleet_lock import FleetLease, LeaseOutcome


class LeaseHeartbeatError(RuntimeError):
    """Raised when the authoritative fleet lease is missing or cannot be renewed."""


class LeaseStore(Protocol):
    """Minimal lease operations used by the heartbeat."""

    def read(self) -> FleetLease:
        """Return the authoritative lease."""

        ...

    def extend(self, *, ttl_s: float | None = None) -> LeaseOutcome:
        """Extend the held lease."""

        ...


class AuthoritativeLeaseHeartbeat:
    """Renew a held lease and verify the remote record after every write."""

    def __init__(
        self,
        store: LeaseStore,
        *,
        holder: str,
        ttl_s: float,
        interval_s: float,
        on_verified_expiry: Callable[[datetime], None] | None = None,
    ) -> None:
        self._store = store
        self._holder = holder
        self._ttl_s = ttl_s
        self._interval_s = interval_s
        self._on_verified_expiry = on_verified_expiry
        self._stop = threading.Event()
        self._failure: LeaseHeartbeatError | None = None
        self._thread: threading.Thread | None = None

    def verify_current(self) -> FleetLease:
        """Reread and validate the authoritative held lease."""

        lease = self._store.read()
        now = datetime.now(UTC)
        if (
            lease.state != "held"
            or lease.holder != self._holder
            or lease.is_expired(now)
        ):
            raise LeaseHeartbeatError(
                "authoritative fleet lease is not held by this qualification"
            )
        expiry = lease.expiry()
        if expiry is None:
            raise LeaseHeartbeatError("authoritative fleet lease has no valid expiry")
        if self._on_verified_expiry is not None:
            self._on_verified_expiry(expiry)
        return lease

    def renew_once(self) -> FleetLease:
        """Extend the lease, then reread and verify its authoritative expiry."""

        outcome = self._store.extend(ttl_s=self._ttl_s)
        if not outcome.ok:
            raise LeaseHeartbeatError(f"fleet lease renewal failed: {outcome.message}")
        authoritative = self.verify_current()
        written_expiry = outcome.lease.expiry()
        authoritative_expiry = authoritative.expiry()
        if (
            written_expiry is None
            or authoritative_expiry is None
            or authoritative_expiry < written_expiry
        ):
            raise LeaseHeartbeatError(
                "authoritative lease expiry did not reflect the renewal"
            )
        return authoritative

    def start(self) -> None:
        """Verify the acquired lease and start its background renewal loop."""

        if self._thread is not None:
            raise RuntimeError("lease heartbeat already started")
        self.verify_current()
        self._thread = threading.Thread(
            target=self._run,
            name="fresh-install-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the renewal loop without releasing the fleet lease."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(30.0, max(1.0, self._interval_s + 1.0)))
            if self._thread.is_alive() and self._failure is None:
                self._failure = LeaseHeartbeatError(
                    "fleet lease heartbeat did not stop before the release boundary"
                )

    def raise_if_failed(self) -> None:
        """Surface a background renewal failure at lifecycle boundaries."""

        if self._failure is not None:
            raise self._failure

    def emergency_extend(self, *, ttl_s: float) -> FleetLease:
        """Make one final verified extension before leaving a failed lease held."""

        outcome = self._store.extend(ttl_s=ttl_s)
        if not outcome.ok:
            raise LeaseHeartbeatError(
                f"emergency fleet lease extension failed: {outcome.message}"
            )
        return self.verify_current()

    def _run(self) -> None:
        """Renew until stopped, retaining the first failure for the main thread."""

        while not self._stop.wait(self._interval_s):
            try:
                self.renew_once()
            except LeaseHeartbeatError as exception:
                self._failure = exception
                self._stop.set()
                return
