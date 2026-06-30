"""SSH and cluster-control primitives for the stability suites.

These functions deliberately keep the *decision* logic (friendly-name mapping,
master identification, SSH command construction) pure and unit-testable, while
isolating the side-effecting parts (``subprocess.run`` over SSH, polling the
live cluster) behind thin wrappers. The crash/relaunch model is faithful: nodes
can be killed and relaunched exactly as an operator would, so the suites observe
real failover, not a graceful shutdown the system could special-case.
"""

from __future__ import annotations

import subprocess
import time

from skulk_test_harness.client import SkulkApiError, SkulkClient
from skulk_test_harness.models import ClusterNode

# Faithful-crash and relaunch shell snippets. ``pkill -9`` is an ungraceful kill
# (no SIGTERM handlers run), which is the worst case the cluster must survive.
KILL_COMMAND = 'pkill -9 -f "uv run skulk"'


def _relaunch_command(repo_path: str) -> str:
    """Return the detached relaunch shell command for a node's repo path.

    Mirrors the operator relaunch ritual: cd into the checkout and start skulk
    under ``nohup`` with output redirected, then ``disown`` so the process
    survives the SSH session closing.
    """

    return (
        f"cd {repo_path} && "
        "nohup uv run skulk -v > ~/skulk-e2e.log 2>&1 & disown"
    )


def _ssh_argv(host: str, command: str) -> list[str]:
    """Build the ``ssh`` argument vector for a remote command.

    Pure helper (no execution) so command construction can be unit-tested. Uses
    a short connect timeout and ``BatchMode`` so a dead/unreachable host fails
    fast and never blocks on an interactive password prompt.
    """

    return [
        "ssh",
        "-o",
        "ConnectTimeout=6",
        "-o",
        "BatchMode=yes",
        host,
        command,
    ]


def _run_ssh(host: str, command: str, *, timeout_s: float = 20.0) -> bool:
    """Execute ``command`` on ``host`` over SSH, returning success.

    Returns ``True`` when ssh exits 0. Any non-zero exit, timeout, or spawn
    failure returns ``False`` rather than raising, because the suites treat an
    unreachable node as a (recorded) assertion failure, not a crash of the
    harness itself.
    """

    try:
        completed = subprocess.run(  # noqa: S603 - argv built from typed config
            _ssh_argv(host, command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def current_master(client: SkulkClient) -> str:
    """Return the current master node ID via ``/v1/diagnostics/node``."""

    return client.get_master_node_id()


def friendly_for_node(client: SkulkClient, node_id: str) -> str:
    """Map a libp2p node ID to its friendly name via cluster state.

    Falls back to the raw node ID when no identity entry is present.
    """

    state = client.get_state()
    return _friendly_for_node_from_state(state, node_id)


def node_for_friendly(client: SkulkClient, friendly: str) -> str | None:
    """Reverse-map a friendly name to its current libp2p node ID, or ``None``."""

    state = client.get_state()
    return _node_for_friendly_from_state(state, friendly)


def _node_identities(state: dict[str, object]) -> dict[str, dict[str, object]]:
    identities = state.get("nodeIdentities")
    if not isinstance(identities, dict):
        return {}
    return {
        str(node_id): identity
        for node_id, identity in identities.items()
        if isinstance(identity, dict)
    }


def _friendly_for_node_from_state(state: dict[str, object], node_id: str) -> str:
    """Pure friendly-name lookup against a state dict (unit-testable)."""

    identity = _node_identities(state).get(node_id)
    if identity is not None:
        friendly = identity.get("friendlyName")
        if isinstance(friendly, str) and friendly:
            return friendly
    return node_id


def _node_for_friendly_from_state(
    state: dict[str, object], friendly: str
) -> str | None:
    """Pure reverse friendly-name lookup against a state dict (unit-testable)."""

    for node_id, identity in _node_identities(state).items():
        if identity.get("friendlyName") == friendly:
            return node_id
    return None


def _present_node_ids(state: dict[str, object]) -> set[str]:
    """Return node IDs the cluster currently considers present (via lastSeen)."""

    last_seen = state.get("lastSeen")
    if not isinstance(last_seen, dict):
        return set()
    return {str(node_id) for node_id in last_seen}


def _kill_command_for_node(node: ClusterNode) -> str:
    return node.kill_command or KILL_COMMAND


def _relaunch_command_for_node(node: ClusterNode) -> str | None:
    if node.relaunch_command:
        return node.relaunch_command
    if node.repo_path:
        return _relaunch_command(node.repo_path)
    return None


def kill_skulk(target: ClusterNode | str) -> bool:
    """Hard-kill the skulk process on ``target`` (faithful crash).

    Returns ``True`` when the kill command ran successfully. Note ``pkill``
    exits 1 when it matched nothing, so a ``False`` here can also mean skulk was
    already down; callers verify the effect via :func:`wait_for_node_absent`.
    """

    if isinstance(target, ClusterNode):
        return _run_ssh(target.ssh_host, _kill_command_for_node(target))
    return _run_ssh(target, KILL_COMMAND)


def relaunch_skulk(node: ClusterNode) -> bool:
    """Relaunch skulk on a node detached over SSH.

    Returns ``True`` when the launch command was accepted by the remote shell.
    Process readiness is confirmed separately via :func:`wait_for_node_present`.
    """

    command = _relaunch_command_for_node(node)
    if command is None:
        return False
    return _run_ssh(node.ssh_host, command)


def _safe_present_node_ids(client: SkulkClient) -> set[str] | None:
    """Fetch present node IDs, returning ``None`` if the API is unreachable.

    The surviving API node may itself briefly error mid-failover; callers retry,
    so transient API errors are swallowed rather than aborting the poll loop.
    """

    try:
        state = client.get_state()
    except (SkulkApiError, OSError):
        return None
    return _present_node_ids(state)


def wait_for_node_absent(
    client: SkulkClient, node_id: str, *, timeout_s: float, poll_interval_s: float = 2.0
) -> bool:
    """Poll cluster state until ``node_id`` drops out of ``lastSeen``."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        present = _safe_present_node_ids(client)
        if present is not None and node_id not in present:
            return True
        time.sleep(poll_interval_s)
    return False


def wait_for_node_present(
    client: SkulkClient, node_id: str, *, timeout_s: float, poll_interval_s: float = 2.0
) -> bool:
    """Poll cluster state until ``node_id`` appears in ``lastSeen``."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        present = _safe_present_node_ids(client)
        if present is not None and node_id in present:
            return True
        time.sleep(poll_interval_s)
    return False


def wait_for_new_master(
    client: SkulkClient,
    old_master_id: str,
    *,
    timeout_s: float,
    poll_interval_s: float = 2.0,
) -> str | None:
    """Poll a surviving node until a new, non-empty master is elected.

    ``client`` must point at a SURVIVING node (not the one being crashed), since
    the killed node's API is gone. Returns the new master node ID, or ``None``
    if no distinct master appears before the timeout.
    """

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            master = client.get_master_node_id()
        except (SkulkApiError, OSError):
            master = ""
        if master and master != old_master_id:
            return master
        time.sleep(poll_interval_s)
    return None
