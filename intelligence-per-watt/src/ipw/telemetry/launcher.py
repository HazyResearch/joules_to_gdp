"""Process management helpers for the bundled energy monitor."""

from __future__ import annotations

import os
import socket
import time
from contextlib import contextmanager
from subprocess import Popen, TimeoutExpired
from typing import Iterator, Mapping, MutableMapping, Sequence

import grpc

from . import _binaries
from .proto import get_stub_bundle

DEFAULT_TARGET = "127.0.0.1:50053"


def normalize_target(target: str) -> str:
    """Normalize a target string into ``host:port`` form."""

    if target.startswith("grpc://"):
        target = target[len("grpc://") :]
    if target.startswith("http://"):
        target = target[len("http://") :]
    if target.startswith("https://"):
        target = target[len("https://") :]
    if "/" in target:
        target = target.split("/", 1)[0]
    if ":" not in target:
        target = f"{target}:50053"
    return target


def wait_for_ready(
    target: str = DEFAULT_TARGET, *, timeout: float = 5.0,
    require_energy_components: bool = False,
) -> bool:
    """Return True when the energy monitor responds to a health check."""

    normalized = normalize_target(target)
    bundle = get_stub_bundle()
    channel = grpc.insecure_channel(normalized)
    stub = bundle.stub_factory(channel)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        response = stub.Health(
            bundle.HealthRequestCls(require_energy_components=require_energy_components),
            timeout=timeout,
        )
        return bool(response.healthy) and (
            not require_energy_components
            or all((response.gpu_energy_available, response.cpu_energy_available,
                    response.dram_energy_available))
        )
    except Exception:
        return False
    finally:
        channel.close()


def _launch_monitor_process(
    args: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = 5.0,
    target: str = DEFAULT_TARGET,
) -> tuple[Popen, str]:
    """Launch the energy monitor and wait for it to become ready.

    Returns a tuple of ``(process, target)``. Raises ``RuntimeError`` if readiness
    is not achieved within ``timeout`` seconds.
    """

    normalized = normalize_target(target)
    host, port = normalized.rsplit(":", 1)

    launch_args = list(args) if args is not None else []

    def _flag_value(flag: str) -> str | None:
        for index, arg in enumerate(launch_args):
            if arg == flag:
                if index + 1 >= len(launch_args):
                    raise ValueError(f"{flag} requires a value")
                return launch_args[index + 1]
            if arg.startswith(f"{flag}="):
                return arg.split("=", 1)[1]
        return None

    requested_port = _flag_value("--port")
    requested_host = _flag_value("--bind-address")
    if requested_port is not None and requested_port != port:
        raise ValueError("Energy monitor --port differs from target")
    if requested_host is not None and requested_host != host:
        raise ValueError("Energy monitor --bind-address differs from target")
    if requested_port is None:
        launch_args.extend(["--port", port])
    if requested_host is None:
        launch_args.extend(["--bind-address", host])

    process = _binaries.launch("energy-monitor", launch_args, env=env)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            process.wait()
            raise RuntimeError(
                f"Energy monitor exited before readiness (exit code {exit_code}); "
                "check the monitor logs and hardware collector"
            )
        if wait_for_ready(normalized, timeout=0.5) and process.poll() is None:
            return process, normalized
        time.sleep(0.25)

    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=2.0)
    except TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)
    raise RuntimeError(
        "Energy monitor failed to become healthy; check the selected GPU collector "
        "and its permissions"
    )


def launch_monitor(
    args: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = 5.0,
    target: str = DEFAULT_TARGET,
) -> tuple[int, str]:
    """Launch a monitor and return its PID and target for manual management."""
    process, normalized = _launch_monitor_process(args, env=env, timeout=timeout, target=target)
    return process.pid, normalized


def _require_free_loopback_target(target: str) -> None:
    host, port_text = target.rsplit(":", 1)
    if host != "127.0.0.1":
        raise RuntimeError("Owned energy monitor requires a 127.0.0.1 target")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise RuntimeError("Energy monitor target has an invalid port") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("Energy monitor target port must be between 1 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"Owned energy monitor target {target} is already in use") from exc


@contextmanager
def ensure_monitor(
    target: str = DEFAULT_TARGET,
    *,
    timeout: float = 5.0,
    launch: bool = True,
    launch_args: Sequence[str] | None = None,
    env: MutableMapping[str, str] | None = None,
) -> Iterator[str]:
    """Ensure a monitor is reachable, optionally launching one in the background.

    Yields the normalized target string. If ``launch`` is True, a background
    process will be spawned when the target is unavailable. The caller is
    responsible for any teardown when the context exits.
    """

    normalized = normalize_target(target)
    process: Popen | None = None
    env_map: Mapping[str, str] = env if env is not None else os.environ
    if env_map.get("IPW_REQUIRE_OWNED_MONITOR") == "1":
        if not launch:
            raise RuntimeError("Owned energy monitor mode requires launch=True")
        _require_free_loopback_target(normalized)
        process, normalized = _launch_monitor_process(
            launch_args, env=env_map, timeout=timeout, target=normalized
        )
    elif not wait_for_ready(normalized, timeout=timeout) and launch:
        process, normalized = _launch_monitor_process(
            launch_args, env=env_map, timeout=timeout, target=normalized
        )

    try:
        yield normalized
    finally:
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=5.0)
            except TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


__all__ = [
    "DEFAULT_TARGET",
    "normalize_target",
    "wait_for_ready",
    "launch_monitor",
    "ensure_monitor",
]
