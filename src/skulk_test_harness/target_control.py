"""SSH control, private recovery snapshots, and restoration verification."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from skulk_test_harness.models import FreshInstallTarget


@dataclass(frozen=True)
class OriginalTargetState:
    """Private baseline used to prove restoration after qualification."""

    git_commit: str | None
    git_status: str | None
    config_sha256: dict[str, str]
    process_arguments: list[str]
    service_status: str | None
    api_node_id: str | None
    cluster_node_count: int | None


@dataclass(frozen=True)
class RecoverySnapshot:
    """Two checksummed copies of one private recovery archive."""

    remote_path: str
    remote_sha256: str
    controller_path: Path
    controller_sha256: str
    original: OriginalTargetState


class SshTargetController:
    """Run bounded commands and tunnels against one explicitly selected target."""

    def __init__(self, target: FreshInstallTarget) -> None:
        if target.kind != "physical" or not target.ssh_host:
            raise ValueError("SSH control requires a physical target with ssh_host")
        self.target = target
        destination = target.ssh_host
        if target.ssh_user:
            destination = f"{target.ssh_user}@{destination}"
        self._destination = destination

    def run(
        self,
        command: str,
        *,
        timeout_s: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Run one remote shell command without exposing it in reports."""

        return subprocess.run(
            [*self._ssh_prefix(), self._destination, command],
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    def start(
        self,
        command: str,
        *,
        log_path: Path,
    ) -> tuple[subprocess.Popen[bytes], BinaryIO]:
        """Start a remote foreground process whose SSH session owns its lifetime."""

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("wb")
        log_path.chmod(0o600)
        process = subprocess.Popen(
            [*self._ssh_prefix(), self._destination, command],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        return process, log_handle

    def open_tunnel(self, *, remote_port: int) -> tuple[int, subprocess.Popen[bytes]]:
        """Open a loopback-only SSH tunnel to the target's default API port."""

        local_port = _available_local_port()
        process = subprocess.Popen(
            [
                *self._ssh_prefix(),
                "-o",
                "ExitOnForwardFailure=yes",
                "-N",
                "-L",
                f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
                self._destination,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.5)
        if process.poll() is not None:
            stderr = (
                process.stderr.read().decode(errors="replace")
                if process.stderr is not None
                else ""
            )
            raise RuntimeError(f"SSH tunnel failed to start: {stderr.strip()}")
        return local_port, process

    def copy_from(self, remote_path: str, local_path: Path) -> None:
        """Copy one private target artifact to the controller."""

        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                *self._scp_prefix(),
                f"{self._destination}:{remote_path}",
                str(local_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        local_path.chmod(0o600)

    def capture_recovery_snapshot(
        self,
        *,
        qualification_id: str,
        controller_root: Path,
        retention_days: int,
        api_node_id: str | None,
        cluster_node_count: int | None,
        api_diagnostics: dict[str, object],
    ) -> RecoverySnapshot:
        """Create mode-600 target and controller archives before stopping Skulk."""

        controller_root = controller_root.expanduser()
        _purge_controller_snapshots(
            controller_root,
            retention_days=retention_days,
        )
        original = self.capture_original_state(
            api_node_id=api_node_id,
            cluster_node_count=cluster_node_count,
        )
        payload = {
            "git_commit": original.git_commit,
            "git_status": original.git_status,
            "config_sha256": original.config_sha256,
            "process_arguments": original.process_arguments,
            "service_status": original.service_status,
            "api_node_id": original.api_node_id,
            "cluster_node_count": original.cluster_node_count,
            "api_diagnostics": api_diagnostics,
        }
        encoded_manifest = _base64_json(payload)
        config_paths = json.dumps(self.target.original_config_paths)
        remote_command = _snapshot_command(
            qualification_id=qualification_id,
            encoded_manifest=encoded_manifest,
            config_paths=config_paths,
            retention_days=retention_days,
        )
        result = self.run(remote_command, timeout_s=120)
        response = json.loads(result.stdout)
        remote_path = response.get("path")
        remote_sha256 = response.get("sha256")
        if not isinstance(remote_path, str) or not isinstance(remote_sha256, str):
            raise RuntimeError("target recovery snapshot returned invalid metadata")
        controller_path = controller_root / qualification_id / "recovery.tar.gz"
        self.copy_from(remote_path, controller_path)
        controller_sha256 = _sha256_file(controller_path)
        if controller_sha256 != remote_sha256:
            raise RuntimeError("controller recovery snapshot checksum mismatch")
        return RecoverySnapshot(
            remote_path=remote_path,
            remote_sha256=remote_sha256,
            controller_path=controller_path,
            controller_sha256=controller_sha256,
            original=original,
        )

    def capture_original_state(
        self,
        *,
        api_node_id: str | None,
        cluster_node_count: int | None,
    ) -> OriginalTargetState:
        """Capture process, service, config, and git state without mutation."""

        git_commit: str | None = None
        git_status: str | None = None
        if self.target.original_checkout:
            quoted_checkout = shlex.quote(self.target.original_checkout)
            result = self.run(
                f"git -C {quoted_checkout} rev-parse HEAD",
                check=False,
                timeout_s=20,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip() or None
            status = self.run(
                f"git -C {quoted_checkout} status --porcelain=v1 --branch",
                check=False,
                timeout_s=20,
            )
            if status.returncode == 0:
                git_status = status.stdout
        config_hashes: dict[str, str] = {}
        for path in self.target.original_config_paths:
            quoted_path = shlex.quote(path)
            result = self.run(
                "if [ -f "
                f"{quoted_path}"
                " ]; then "
                "if command -v shasum >/dev/null 2>&1; "
                f"then shasum -a 256 {quoted_path}; "
                f"else sha256sum {quoted_path}; fi; fi",
                check=False,
                timeout_s=20,
            )
            if result.returncode == 0 and result.stdout.strip():
                config_hashes[path] = result.stdout.split()[0]
        process_result = self.run(
            "ps ax -o command= | "
            "grep -E '(^|[[:space:]/])(uv[[:space:]]+run[[:space:]]+)?"
            "[s]kulk([[:space:]]|$)' || true",
            timeout_s=20,
        )
        process_arguments = sorted(
            line.strip() for line in process_result.stdout.splitlines() if line.strip()
        )
        service_status: str | None = None
        if self.target.service_status_command:
            status_result = self.run(
                self.target.service_status_command,
                check=False,
                timeout_s=30,
            )
            service_status = (
                f"exit={status_result.returncode}\n"
                f"{status_result.stdout}{status_result.stderr}"
            )
        return OriginalTargetState(
            git_commit=git_commit,
            git_status=git_status,
            config_sha256=config_hashes,
            process_arguments=process_arguments,
            service_status=service_status,
            api_node_id=api_node_id,
            cluster_node_count=cluster_node_count,
        )

    def verify_restored_state(
        self,
        original: OriginalTargetState,
        *,
        api_node_id: str | None,
        cluster_node_count: int | None,
    ) -> list[str]:
        """Return every restoration mismatch; an empty list proves restoration."""

        restored = self.capture_original_state(
            api_node_id=api_node_id,
            cluster_node_count=cluster_node_count,
        )
        failures: list[str] = []
        if restored.git_commit != original.git_commit:
            failures.append("original checkout commit changed")
        if restored.git_status != original.git_status:
            failures.append("original checkout status changed")
        if restored.config_sha256 != original.config_sha256:
            failures.append("original configuration hash changed")
        if restored.process_arguments != original.process_arguments:
            failures.append("original process arguments were not restored")
        if original.api_node_id and restored.api_node_id != original.api_node_id:
            failures.append("original API identity was not restored")
        if (
            original.cluster_node_count is not None
            and restored.cluster_node_count is not None
            and restored.cluster_node_count < original.cluster_node_count
        ):
            failures.append("original fleet membership did not rejoin")
        if self.target.service_status_command and restored.service_status is None:
            failures.append("restored service status could not be read")
        if (
            original.service_status is not None
            and restored.service_status is not None
            and original.service_status.splitlines()[0]
            != restored.service_status.splitlines()[0]
        ):
            failures.append("original service manager state was not restored")
        return failures

    def _ssh_prefix(self) -> list[str]:
        prefix = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-p",
            str(self.target.ssh_port),
        ]
        if self.target.ssh_identity_file is not None:
            prefix.extend(["-i", str(self.target.ssh_identity_file.expanduser())])
        return prefix

    def _scp_prefix(self) -> list[str]:
        prefix = [
            "scp",
            "-q",
            "-P",
            str(self.target.ssh_port),
        ]
        if self.target.ssh_identity_file is not None:
            prefix.extend(["-i", str(self.target.ssh_identity_file.expanduser())])
        return prefix


def _available_local_port() -> int:
    """Reserve an ephemeral loopback port number for an immediate SSH tunnel."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _base64_json(payload: object) -> str:
    """Encode private JSON for a shell-safe remote Python argument."""

    import base64

    raw = json.dumps(payload, sort_keys=True).encode()
    return base64.b64encode(raw).decode("ascii")


def _snapshot_command(
    *,
    qualification_id: str,
    encoded_manifest: str,
    config_paths: str,
    retention_days: int,
) -> str:
    """Build the portable remote archive command."""

    script = (
        "import base64,hashlib,json,os,pathlib,shutil,tarfile,time;"
        "qid=os.environ['QID'];"
        "root=pathlib.Path.home()/'.local/state/skulk-test-harness/recovery';"
        "root.mkdir(parents=True,exist_ok=True,mode=0o700);"
        "cutoff=time.time()-int(os.environ['RETENTION'])*86400;"
        "[(p.unlink()) for p in root.glob('*.tar.gz') if p.stat().st_mtime<cutoff];"
        "stage=root/(qid+'.stage');"
        "shutil.rmtree(stage,ignore_errors=True);stage.mkdir(mode=0o700);"
        "(stage/'manifest.json').write_bytes(base64.b64decode(os.environ['MANIFEST']));"
        "paths=json.loads(os.environ['CONFIG_PATHS']);"
        "[(shutil.copy2(path,stage/('config-'+str(index)))) "
        "for index,path in enumerate(paths) if pathlib.Path(path).is_file()];"
        "archive=root/(qid+'.tar.gz');"
        "tar=tarfile.open(archive,'w:gz');tar.add(stage,arcname='recovery');tar.close();"
        "os.chmod(archive,0o600);shutil.rmtree(stage);"
        "digest=hashlib.sha256(archive.read_bytes()).hexdigest();"
        "print(json.dumps({'path':str(archive),'sha256':digest}))"
    )
    environment = (
        f"QID={shlex.quote(qualification_id)} "
        f"RETENTION={retention_days} "
        f"MANIFEST={shlex.quote(encoded_manifest)} "
        f"CONFIG_PATHS={shlex.quote(config_paths)}"
    )
    return f"{environment} python3 -c {shlex.quote(script)}"


def _sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _purge_controller_snapshots(root: Path, *, retention_days: int) -> None:
    """Remove controller recovery directories older than the retention policy."""

    if not root.exists():
        return
    cutoff = time.time() - retention_days * 86400
    for child in root.iterdir():
        if not child.is_dir() or child.stat().st_mtime >= cutoff:
            continue
        shutil.rmtree(child)
