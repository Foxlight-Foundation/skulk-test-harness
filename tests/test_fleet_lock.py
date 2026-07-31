"""Tests for the git-backed fleet lease.

These drive the real ``FleetLockStore`` against a genuine local bare git repo
(the "remote"), so the compare-and-swap, TTL expiry, and holder rules are
exercised through actual git pushes rather than mocks.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

import skulk_test_harness.fleet_lock as fleet_lock_module
from skulk_test_harness.fleet_lock import FleetLease, FleetLockStore, LeaseOutcome
from skulk_test_harness.models import FleetLock, HarnessConfig

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required for fleet-lock tests"
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A bare git repo seeded with a free lock on ``main``."""

    remote_dir = tmp_path / "remote.git"
    subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "init", "--quiet", "--bare", str(remote_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    seed = tmp_path / "seed"
    subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "clone", "--quiet", str(remote_dir), str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(seed, "checkout", "-b", "main")
    lock_dir = seed / "coordination"
    lock_dir.mkdir(parents=True)
    (lock_dir / "fleet-lock.json").write_text(FleetLease().to_json())
    _git(seed, "add", "-A")
    _git(seed, "commit", "--quiet", "-m", "seed")
    _git(seed, "push", "--quiet", "-u", "origin", "main")
    return remote_dir


def _store(remote_dir: Path, cache_dir: Path, holder: str) -> FleetLockStore:
    return FleetLockStore(
        FleetLock(remote=str(remote_dir), holder=holder, cache_dir=cache_dir)
    )


def test_acquire_then_release(remote: Path, tmp_path: Path) -> None:
    store = _store(remote, tmp_path / "claude", "claude")
    acquired = store.acquire(branch="feature/x", host="devbox")
    assert acquired.ok
    lease = store.read()
    assert lease.state == "held"
    assert lease.holder == "claude"
    assert lease.branch == "feature/x"

    released = store.release()
    assert released.ok
    assert store.read().state == "free"


def test_second_agent_is_refused(remote: Path, tmp_path: Path) -> None:
    claude = _store(remote, tmp_path / "claude", "claude")
    codex = _store(remote, tmp_path / "codex", "codex")
    assert claude.acquire(branch="feature/x", host="a").ok

    refused = codex.acquire(branch="feature/y", host="b")
    assert not refused.ok
    assert "claude" in refused.message
    # The refusal did not change the holder.
    assert codex.read().holder == "claude"


def test_expired_lock_is_reclaimable(remote: Path, tmp_path: Path) -> None:
    claude = _store(remote, tmp_path / "claude", "claude")
    codex = _store(remote, tmp_path / "codex", "codex")
    # A negative TTL makes the lease already expired the moment it is written.
    assert claude.acquire(branch="feature/x", host="a", ttl_s=-1).ok

    taken = codex.acquire(branch="feature/y", host="b")
    assert taken.ok
    assert "stale" in taken.message
    assert codex.read().holder == "codex"


def test_extend_requires_holder(remote: Path, tmp_path: Path) -> None:
    claude = _store(remote, tmp_path / "claude", "claude")
    codex = _store(remote, tmp_path / "codex", "codex")
    assert claude.acquire(branch="feature/x", host="a").ok

    # A non-holder cannot extend.
    assert not codex.extend().ok
    # The holder can.
    assert claude.extend(ttl_s=60).ok


def test_force_release_breaks_foreign_lock(remote: Path, tmp_path: Path) -> None:
    claude = _store(remote, tmp_path / "claude", "claude")
    codex = _store(remote, tmp_path / "codex", "codex")
    assert claude.acquire(branch="feature/x", host="a").ok

    assert not codex.release().ok  # not the holder, no force
    forced = codex.release(force=True)
    assert forced.ok
    assert codex.read().state == "free"


def test_commit_and_push_loses_cas_when_remote_advances(
    remote: Path, tmp_path: Path
) -> None:
    claude = _store(remote, tmp_path / "claude", "claude")
    codex = _store(remote, tmp_path / "codex", "codex")
    # Sync claude's clone to the current tip.
    claude.read()
    # codex advances the remote out from under claude.
    assert codex.acquire(branch="feature/y", host="b").ok
    # claude's push from a stale base must be rejected (the CAS lost).
    lost = claude._commit_and_push(  # pyright: ignore[reportPrivateUsage]
        FleetLease(state="held", holder="claude"), "claude tries anyway"
    )
    assert lost is False
    # The remote still reflects codex's claim.
    assert claude.read().holder == "codex"


def test_concurrent_read_cannot_erase_pending_release(
    remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status reader must not reset the shared clone during a lease write."""

    cache_dir = tmp_path / "shared-cache"
    writer = _store(remote, cache_dir, "claude")
    reader = _store(remote, cache_dir, "claude")
    assert writer.acquire(branch="feature/x", host="devbox").ok

    writer_reached_status = threading.Event()
    allow_writer_to_continue = threading.Event()
    reader_finished = threading.Event()
    original_git = writer._git  # pyright: ignore[reportPrivateUsage]

    def pause_before_release_status(
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if args == ("status", "--porcelain"):
            writer_reached_status.set()
            assert allow_writer_to_continue.wait(timeout=5)
        return original_git(*args, check=check)

    monkeypatch.setattr(writer, "_git", pause_before_release_status)
    release_outcomes: list[LeaseOutcome] = []
    observed_leases: list[FleetLease] = []

    def release() -> None:
        release_outcomes.append(writer.release())

    def read() -> None:
        observed_leases.append(reader.read())
        reader_finished.set()

    writer_thread = threading.Thread(target=release)
    reader_thread = threading.Thread(target=read)
    writer_thread.start()
    assert writer_reached_status.wait(timeout=5)
    reader_thread.start()
    time.sleep(0.05)
    assert not reader_finished.is_set()

    allow_writer_to_continue.set()
    writer_thread.join(timeout=5)
    reader_thread.join(timeout=5)

    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert release_outcomes and release_outcomes[0].ok
    assert observed_leases == [FleetLease()]


def test_clobbered_no_op_write_fails_authoritative_verification(
    remote: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older unsynchronized reader cannot create a false-successful release."""

    store = _store(remote, tmp_path / "cache", "claude")
    assert store.acquire(branch="feature/x", host="devbox").ok
    original_git = store._git  # pyright: ignore[reportPrivateUsage]
    clobbered = False

    def clobber_before_status(
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal clobbered
        if args == ("status", "--porcelain") and not clobbered:
            clobbered = True
            original_git("reset", "--quiet", "--hard", "origin/main")
        return original_git(*args, check=check)

    monkeypatch.setattr(store, "_git", clobber_before_status)

    released = store._commit_and_push(  # pyright: ignore[reportPrivateUsage]
        FleetLease(),
        "release clobbered by old reader",
    )

    assert released is False
    assert store.read().state == "held"


def test_disabled_when_unconfigured() -> None:
    cfg = HarnessConfig()
    assert cfg.fleet_lock is None


def test_git_coordination_operations_have_a_hard_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A black-holed coordination remote must not wedge the lease heartbeat."""

    observed_timeouts: list[float] = []

    def bounded_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        timeout = kwargs.get("timeout")
        assert isinstance(timeout, (int, float))
        observed_timeouts.append(float(timeout))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(fleet_lock_module.subprocess, "run", bounded_run)
    existing_cache = tmp_path / "existing"
    configured = FleetLock(
        remote="https://example.invalid/coordination.git",
        holder="codex",
        cache_dir=existing_cache,
    )
    store = FleetLockStore(configured)
    existing_cache.mkdir(parents=True)
    (existing_cache / ".git").mkdir()
    store._git("status")  # pyright: ignore[reportPrivateUsage]

    clone_store = FleetLockStore(
        configured.model_copy(update={"cache_dir": tmp_path / "new-clone"})
    )
    clone_store._ensure_clone()  # pyright: ignore[reportPrivateUsage]

    assert observed_timeouts == [30.0, 30.0]


def test_require_fleet_or_refuse(remote: Path, tmp_path: Path) -> None:
    import typer

    from skulk_test_harness import cli

    claude = _store(remote, tmp_path / "claude", "claude")
    assert claude.acquire(branch="feature/x", host="a").ok

    codex_cfg = HarnessConfig(
        fleet_lock=FleetLock(
            remote=str(remote), holder="codex", cache_dir=tmp_path / "codex-pf"
        )
    )
    # A foreign holder blocks the execute path...
    with pytest.raises(typer.Exit):
        cli._require_fleet_or_refuse(codex_cfg, force=False)  # pyright: ignore[reportPrivateUsage]
    # ...unless forced.
    cli._require_fleet_or_refuse(codex_cfg, force=True)  # pyright: ignore[reportPrivateUsage]

    # The holder itself is never blocked.
    claude_cfg = HarnessConfig(
        fleet_lock=FleetLock(
            remote=str(remote), holder="claude", cache_dir=tmp_path / "claude-pf"
        )
    )
    cli._require_fleet_or_refuse(claude_cfg, force=False)  # pyright: ignore[reportPrivateUsage]

    # Unconfigured => never blocks.
    cli._require_fleet_or_refuse(HarnessConfig(), force=False)  # pyright: ignore[reportPrivateUsage]


def test_fleet_cli_round_trip(remote: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from skulk_test_harness import cli

    cfg_file = tmp_path / "skulk-harness.yaml"
    cache = tmp_path / "cli-clone"
    cfg_file.write_text(
        "fleet_lock:\n"
        f"  remote: {remote}\n"
        "  holder: claude\n"
        f"  cache_dir: {cache}\n"
    )
    runner = CliRunner()

    acquired = runner.invoke(
        cli.app, ["fleet", "acquire", "--branch", "feature/x", "--config", str(cfg_file)]
    )
    assert acquired.exit_code == 0, acquired.output
    assert "acquired the fleet" in acquired.output

    status = runner.invoke(cli.app, ["fleet", "status", "--config", str(cfg_file)])
    assert status.exit_code == 0
    assert "HELD" in status.output

    released = runner.invoke(cli.app, ["fleet", "release", "--config", str(cfg_file)])
    assert released.exit_code == 0
    assert "released the fleet" in released.output
