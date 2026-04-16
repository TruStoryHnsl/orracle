"""Stateless ComfyUI launcher — spawn/kill per job.

When a machine is configured with `comfyui_mode: stateless`, this module
SSH-spawns a fresh ComfyUI process on a random free port before each job
and kills the entire process group on completion or failure.

Usage:
    launcher = ComfyUILauncher(machine_cfg)
    url = launcher.start()   # blocks until ComfyUI is ready
    try:
        ... run job using url ...
    finally:
        launcher.stop()
"""

from __future__ import annotations

import logging
import random
import subprocess
import time

from shared import get_machine

log = logging.getLogger(__name__)

_SSH_OPTS = [
    '-o', 'ConnectTimeout=10',
    '-o', 'BatchMode=yes',
    '-o', 'StrictHostKeyChecking=accept-new',
]

# Port range for ephemeral ComfyUI instances
EPHEMERAL_PORT_MIN = 18100
EPHEMERAL_PORT_MAX = 18199

# How long to wait for ComfyUI to come up
STARTUP_TIMEOUT = 120  # seconds


class ComfyUILauncher:
    """Spawn and kill a stateless ComfyUI instance on a remote machine.

    Thread-safe: start()/stop() can be called from worker threads.
    """

    def __init__(self, machine_name: str):
        self._machine_name = machine_name
        self._port: int | None = None
        self._pid: int | None = None
        self._pgid: int | None = None
        self._url: str | None = None

    def start(self) -> str | None:
        """Spawn ComfyUI on a random free port. Returns URL or None on failure."""
        machine = get_machine(self._machine_name)
        if not machine:
            log.error('ComfyUILauncher: machine %s not found', self._machine_name)
            return None

        host = machine.get('host', '')
        user = machine.get('user', '')
        target = f'{user}@{host}' if user else host

        port = self._pick_port(target)
        if not port:
            log.error('ComfyUILauncher: no free port found on %s', self._machine_name)
            return None

        self._port = port
        comfy_dir = machine.get('comfyui_dir', '~/comfy/ComfyUI')

        # Launch ComfyUI as its own process group, capture PID+PGID
        launch_cmd = (
            f'cd {comfy_dir} && '
            f'setsid python main.py --listen 0.0.0.0 --port {port} '
            f'--disable-auto-launch --disable-xformers '
            f'>/tmp/comfy_stateless_{port}.log 2>&1 & '
            f'echo PID=$! PGID=$(ps -o pgid= -p $! | tr -d " ")'
        )

        try:
            r = subprocess.run(
                ['ssh'] + _SSH_OPTS + [target, f'bash -c {repr(launch_cmd)}'],
                capture_output=True, text=True, timeout=20,
            )
        except subprocess.TimeoutExpired:
            log.error('ComfyUILauncher: SSH timeout launching on %s', self._machine_name)
            return None

        if r.returncode != 0:
            log.error('ComfyUILauncher: launch failed on %s: %s',
                      self._machine_name, r.stderr.strip())
            return None

        # Parse PID and PGID from output
        pid = pgid = None
        for token in r.stdout.split():
            if token.startswith('PID='):
                try:
                    pid = int(token[4:])
                except ValueError:
                    pass
            elif token.startswith('PGID='):
                try:
                    pgid = int(token[5:])
                except ValueError:
                    pass

        if not pid:
            log.error('ComfyUILauncher: could not parse PID from output: %s', r.stdout)
            return None

        self._pid = pid
        self._pgid = pgid or pid
        url = f'http://{host}:{port}'
        self._url = url

        log.info('ComfyUILauncher: launched PID=%d PGID=%d on %s port %d',
                 pid, self._pgid, self._machine_name, port)

        # Wait for ComfyUI to come up
        if not self._wait_ready(target, host, port):
            log.error('ComfyUILauncher: ComfyUI did not start within %ds', STARTUP_TIMEOUT)
            self.stop()
            return None

        return url

    def stop(self):
        """Kill the ComfyUI process group on the remote machine."""
        if not self._pgid and not self._pid:
            return

        machine = get_machine(self._machine_name)
        if not machine:
            return

        host = machine.get('host', '')
        user = machine.get('user', '')
        target = f'{user}@{host}' if user else host

        kill_target = self._pgid or self._pid
        signal_flag = '-TERM'

        # Kill the process group to catch all child processes
        kill_cmd = f'kill {signal_flag} -{kill_target} 2>/dev/null || kill {signal_flag} {self._pid} 2>/dev/null; echo done'

        try:
            subprocess.run(
                ['ssh'] + _SSH_OPTS + [target, f'bash -c {repr(kill_cmd)}'],
                capture_output=True, text=True, timeout=10,
            )
            log.info('ComfyUILauncher: killed PGID=%d on %s', kill_target, self._machine_name)
        except Exception:
            log.exception('ComfyUILauncher: failed to kill process on %s', self._machine_name)

        self._pid = None
        self._pgid = None
        self._url = None
        self._port = None

    def _pick_port(self, target: str) -> int | None:
        """Find a free port in the ephemeral range on the remote machine."""
        ports = list(range(EPHEMERAL_PORT_MIN, EPHEMERAL_PORT_MAX + 1))
        random.shuffle(ports)

        # Check which ports are in use via ss/netstat
        check_cmd = "ss -tlnH 2>/dev/null | awk '{print $4}' | grep -oE '[0-9]+$' || true"
        try:
            r = subprocess.run(
                ['ssh'] + _SSH_OPTS + [target, check_cmd],
                capture_output=True, text=True, timeout=10,
            )
            used = set()
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    used.add(int(line))

            for p in ports:
                if p not in used:
                    return p
        except Exception:
            pass

        # Fallback: pick a random port from the range
        return random.choice(ports)

    def _wait_ready(self, target: str, host: str, port: int) -> bool:
        """Poll until ComfyUI's HTTP endpoint responds or timeout."""
        import urllib.request
        import urllib.error

        url = f'http://{host}:{port}/system_stats'
        deadline = time.time() + STARTUP_TIMEOUT

        while time.time() < deadline:
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=3):
                    return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(3)

        return False


def spawn_for_job(machine_name: str) -> tuple[str | None, 'ComfyUILauncher']:
    """Convenience: create a launcher, start it, return (url, launcher).

    Caller must call launcher.stop() in a finally block.
    """
    launcher = ComfyUILauncher(machine_name)
    url = launcher.start()
    return url, launcher
