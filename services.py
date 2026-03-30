"""Service lifecycle manager for orracle.

Tracks and controls AI services (ComfyUI, Ollama, SD Forge) across all
registered machines. Provides health monitoring, start/stop via SSH,
and endpoint routing for the job queue.

Services are defined in machines.yaml under each machine's 'services' key:

    machines:
      orrion:
        services:
          comfyui:
            type: comfyui
            url: http://192.168.1.152:8188
            start_cmd: "tmux send-keys -t comfy '...' Enter"
            stop_cmd: "tmux send-keys -t comfy C-c"
          ollama:
            type: ollama
            url: http://192.168.1.152:11434
            start_cmd: "systemctl start ollama"
            stop_cmd: "systemctl stop ollama"
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum

from shared import load_machines, get_machine

log = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'
    STARTING = 'starting'
    STOPPING = 'stopping'
    UNKNOWN = 'unknown'
    ERROR = 'error'


# Health check endpoints per service type
HEALTH_ENDPOINTS = {
    'comfyui': '/system_stats',
    'ollama': '/api/tags',
    'sdforge': '/sdapi/v1/progress',
}


@dataclass
class ServiceState:
    machine: str
    name: str
    stype: str          # service type: comfyui, ollama, sdforge
    url: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    start_cmd: str = ''
    stop_cmd: str = ''
    last_check: float = 0
    last_change: float = 0
    error: str = ''
    meta: dict = field(default_factory=dict)   # type-specific info (models loaded, GPU %, etc.)

    def to_dict(self) -> dict:
        return {
            'machine': self.machine,
            'name': self.name,
            'type': self.stype,
            'url': self.url,
            'status': self.status.value,
            'start_cmd': bool(self.start_cmd),
            'stop_cmd': bool(self.stop_cmd),
            'last_check': self.last_check,
            'last_change': self.last_change,
            'error': self.error,
            'meta': self.meta,
        }


class ServiceManager:
    """Tracks all services across all machines."""

    def __init__(self):
        self._services: dict[tuple[str, str], ServiceState] = {}
        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._listeners: list = []  # callbacks for status changes

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self):
        """Load service definitions from machines.yaml and probe health."""
        machines = load_machines()
        with self._lock:
            # Track which services still exist in config
            seen = set()
            for mname, mcfg in machines.items():
                for sname, scfg in mcfg.get('services', {}).items():
                    key = (mname, sname)
                    seen.add(key)
                    if key not in self._services:
                        self._services[key] = ServiceState(
                            machine=mname,
                            name=sname,
                            stype=scfg.get('type', sname),
                            url=scfg.get('url', ''),
                            start_cmd=scfg.get('start_cmd', ''),
                            stop_cmd=scfg.get('stop_cmd', ''),
                        )
                    else:
                        # Update config fields (URL, commands may change)
                        svc = self._services[key]
                        svc.url = scfg.get('url', svc.url)
                        svc.start_cmd = scfg.get('start_cmd', svc.start_cmd)
                        svc.stop_cmd = scfg.get('stop_cmd', svc.stop_cmd)
                        svc.stype = scfg.get('type', svc.stype)

            # Remove services no longer in config
            for key in list(self._services):
                if key not in seen:
                    del self._services[key]

        # Probe health for all discovered services
        self._check_all()

    def get_all(self) -> list[dict]:
        """Return all service states as dicts."""
        with self._lock:
            return [s.to_dict() for s in self._services.values()]

    def get_service(self, machine: str, service: str) -> dict | None:
        """Get a single service state."""
        with self._lock:
            svc = self._services.get((machine, service))
            return svc.to_dict() if svc else None

    def get_by_type(self, stype: str) -> list[dict]:
        """Get all services of a given type (e.g., 'comfyui')."""
        with self._lock:
            return [s.to_dict() for s in self._services.values()
                    if s.stype == stype]

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    def check_health(self, machine: str, service: str) -> ServiceStatus:
        """Check health of a specific service. Updates internal state."""
        with self._lock:
            svc = self._services.get((machine, service))
            if not svc:
                return ServiceStatus.UNKNOWN

        status = self._probe(svc)

        with self._lock:
            old_status = svc.status
            svc.status = status
            svc.last_check = time.time()
            if status != old_status:
                svc.last_change = time.time()
                svc.error = '' if status == ServiceStatus.ONLINE else svc.error
                self._notify(svc)

        return status

    def _probe(self, svc: ServiceState) -> ServiceStatus:
        """HTTP probe a service endpoint."""
        if not svc.url:
            return ServiceStatus.UNKNOWN

        endpoint = HEALTH_ENDPOINTS.get(svc.stype, '/')
        url = f'{svc.url}{endpoint}'

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                # Extract type-specific metadata
                svc.meta = self._extract_meta(svc.stype, data)
                return ServiceStatus.ONLINE
        except urllib.error.HTTPError as e:
            # A 4xx/5xx still means the service is running
            if e.code < 500:
                return ServiceStatus.ONLINE
            svc.error = f'HTTP {e.code}'
            return ServiceStatus.ERROR
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            svc.error = str(e)
            return ServiceStatus.OFFLINE

    def _extract_meta(self, stype: str, data: dict) -> dict:
        """Extract useful metadata from health check response."""
        if stype == 'comfyui':
            devices = data.get('devices', [])
            if devices:
                d = devices[0]
                return {
                    'vram_total': d.get('vram_total', 0),
                    'vram_free': d.get('vram_free', 0),
                    'torch_vram': d.get('torch_vram_total', 0),
                }
            return {}
        elif stype == 'ollama':
            models = data.get('models', [])
            return {
                'model_count': len(models),
                'models': [m['name'] for m in models[:5]],
            }
        return {}

    def _check_all(self):
        """Probe all services (called from discover and monitor thread)."""
        with self._lock:
            services = list(self._services.items())

        for (machine, service), _ in services:
            self.check_health(machine, service)

    # ------------------------------------------------------------------
    # Service control
    # ------------------------------------------------------------------

    def start_service(self, machine: str, service: str) -> dict:
        """Start a service on a remote machine via SSH."""
        with self._lock:
            svc = self._services.get((machine, service))
            if not svc:
                return {'ok': False, 'error': 'Service not found'}
            if not svc.start_cmd:
                return {'ok': False, 'error': 'No start_cmd configured'}
            if svc.status == ServiceStatus.ONLINE:
                return {'ok': True, 'message': 'Already online'}

            svc.status = ServiceStatus.STARTING
            svc.last_change = time.time()
            start_cmd = svc.start_cmd
            self._notify(svc)

        machines = load_machines()
        mcfg = machines.get(machine, {})
        host = mcfg.get('host', '')
        user = mcfg.get('user', '')

        if not host:
            return {'ok': False, 'error': 'Machine host not configured'}

        # Run start command via SSH (or locally)
        try:
            if mcfg.get('is_local'):
                subprocess.Popen(start_cmd, shell=True,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            else:
                ssh_target = f'{user}@{host}' if user else host
                subprocess.Popen(
                    ['ssh', '-o', 'ConnectTimeout=10', ssh_target, start_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            with self._lock:
                svc.status = ServiceStatus.ERROR
                svc.error = str(e)
            return {'ok': False, 'error': str(e)}

        # Start polling for the service to come online
        threading.Thread(
            target=self._wait_for_online,
            args=(machine, service, 120),  # 2 min timeout
            daemon=True,
        ).start()

        return {'ok': True, 'message': 'Starting...'}

    def stop_service(self, machine: str, service: str) -> dict:
        """Stop a service on a remote machine via SSH."""
        with self._lock:
            svc = self._services.get((machine, service))
            if not svc:
                return {'ok': False, 'error': 'Service not found'}
            if not svc.stop_cmd:
                return {'ok': False, 'error': 'No stop_cmd configured'}

            svc.status = ServiceStatus.STOPPING
            svc.last_change = time.time()
            stop_cmd = svc.stop_cmd
            self._notify(svc)

        machines = load_machines()
        mcfg = machines.get(machine, {})
        host = mcfg.get('host', '')
        user = mcfg.get('user', '')

        try:
            if mcfg.get('is_local'):
                subprocess.run(stop_cmd, shell=True, timeout=10,
                               capture_output=True)
            else:
                ssh_target = f'{user}@{host}' if user else host
                subprocess.run(
                    ['ssh', '-o', 'ConnectTimeout=10', ssh_target, stop_cmd],
                    timeout=15, capture_output=True,
                )
        except Exception as e:
            with self._lock:
                svc.status = ServiceStatus.ERROR
                svc.error = str(e)
            return {'ok': False, 'error': str(e)}

        # Update status after stop
        time.sleep(2)
        self.check_health(machine, service)
        return {'ok': True}

    def _wait_for_online(self, machine: str, service: str, timeout: int):
        """Poll until a service comes online or times out."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.check_health(machine, service)
            if status == ServiceStatus.ONLINE:
                return
            time.sleep(3)

        # Timed out
        with self._lock:
            svc = self._services.get((machine, service))
            if svc and svc.status == ServiceStatus.STARTING:
                svc.status = ServiceStatus.ERROR
                svc.error = f'Start timed out after {timeout}s'
                self._notify(svc)

    # ------------------------------------------------------------------
    # Routing helpers (used by job queue)
    # ------------------------------------------------------------------

    def find_online(self, stype: str) -> list[dict]:
        """Find all online services of a given type."""
        with self._lock:
            return [s.to_dict() for s in self._services.values()
                    if s.stype == stype and s.status == ServiceStatus.ONLINE]

    def is_healthy(self, machine: str, required_types: list[str]) -> bool:
        """Check if a machine has all required service types online."""
        with self._lock:
            for stype in required_types:
                found = False
                for (m, _), svc in self._services.items():
                    if m == machine and svc.stype == stype and \
                       svc.status == ServiceStatus.ONLINE:
                        found = True
                        break
                if not found:
                    return False
            return True

    def get_endpoint(self, machine: str, stype: str) -> str | None:
        """Get the URL of a specific service type on a machine."""
        with self._lock:
            for (m, _), svc in self._services.items():
                if m == machine and svc.stype == stype and \
                   svc.status == ServiceStatus.ONLINE:
                    return svc.url
            return None

    # ------------------------------------------------------------------
    # Background monitor
    # ------------------------------------------------------------------

    def start_monitor(self, interval: int = 30):
        """Start background health monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True)
        self._monitor_thread.start()
        log.info('Service monitor started (interval=%ds)', interval)

    def stop_monitor(self):
        """Stop background health monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def _monitor_loop(self, interval: int):
        """Periodic health check loop."""
        while not self._stop_event.is_set():
            try:
                self._check_all()
            except Exception:
                log.exception('Error in service health check')
            self._stop_event.wait(interval)

    # ------------------------------------------------------------------
    # Event listeners (for SSE, dashboard updates)
    # ------------------------------------------------------------------

    def add_listener(self, callback):
        """Register a callback for service status changes.

        Callback signature: callback(service_state_dict)
        """
        self._listeners.append(callback)

    def remove_listener(self, callback):
        """Remove a status change listener."""
        self._listeners = [cb for cb in self._listeners if cb is not callback]

    def _notify(self, svc: ServiceState):
        """Notify all listeners of a status change."""
        data = svc.to_dict()
        for cb in self._listeners:
            try:
                cb(data)
            except Exception:
                log.exception('Error in service listener')


# ======================================================================
# Compute Load Watcher
# ======================================================================

@dataclass
class MachineLoad:
    """Snapshot of a machine's compute load."""
    machine: str
    gpu_util: float = 0.0       # 0-100 GPU utilization %
    gpu_mem_used: float = 0.0   # GPU memory used (MB)
    gpu_mem_total: float = 0.0  # GPU memory total (MB)
    cpu_load_1m: float = 0.0    # 1-minute CPU load average
    cpu_count: int = 1          # number of CPU cores
    busy: bool = False          # True if other programs are using significant compute
    ts: float = 0.0             # timestamp of this reading
    error: str = ''

    def to_dict(self) -> dict:
        return {
            'machine': self.machine,
            'gpu_util': self.gpu_util,
            'gpu_mem_used': self.gpu_mem_used,
            'gpu_mem_total': self.gpu_mem_total,
            'cpu_load_1m': self.cpu_load_1m,
            'cpu_count': self.cpu_count,
            'busy': self.busy,
            'ts': self.ts,
            'error': self.error,
        }


class ComputeWatcher:
    """Monitors compute load on remote machines and throttles low-priority jobs.

    When a throttle-enabled job is running on a machine and that machine
    becomes busy with other work, the watcher SIGSTOPs the job.
    When the machine goes idle, it SIGCONTs the job.

    "Busy" is defined as:
    - GPU utilization > gpu_threshold from processes other than the training job
    - OR CPU load average > cpu_threshold * core_count
    """

    def __init__(self, gpu_threshold: float = 30.0, cpu_threshold: float = 0.8):
        self._gpu_threshold = gpu_threshold    # GPU util % that counts as "busy"
        self._cpu_threshold = cpu_threshold    # CPU load / core_count threshold
        self._loads: dict[str, MachineLoad] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._job_queue = None  # set by wire()
        self._listeners: list = []

    def wire(self, job_queue):
        """Connect to the job queue for throttle decisions."""
        self._job_queue = job_queue

    def start(self, interval: int = 15):
        """Start the background load monitoring loop."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(interval,), daemon=True)
        self._thread.start()
        log.info('ComputeWatcher started (interval=%ds, gpu_thresh=%.0f%%, cpu_thresh=%.1f)',
                 interval, self._gpu_threshold, self._cpu_threshold)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_load(self, machine: str) -> dict | None:
        with self._lock:
            ml = self._loads.get(machine)
            return ml.to_dict() if ml else None

    def get_all_loads(self) -> list[dict]:
        with self._lock:
            return [ml.to_dict() for ml in self._loads.values()]

    def add_listener(self, callback):
        self._listeners.append(callback)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self, interval: int):
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                log.exception('ComputeWatcher error')
            self._stop_event.wait(interval)

    def _tick(self):
        """One cycle: poll loads, then throttle/unthrottle jobs."""
        if not self._job_queue:
            return

        # Find machines with throttle-enabled running or auto-suspended jobs
        throttled_jobs = self._get_throttled_jobs()
        if not throttled_jobs:
            return

        machines_needed = set(j['machine'] for j in throttled_jobs if j['machine'])

        # Poll load for each relevant machine
        for machine in machines_needed:
            ml = self._poll_machine(machine)
            with self._lock:
                self._loads[machine] = ml
            self._notify_load(ml)

        # Make throttle decisions
        for job in throttled_jobs:
            machine = job['machine']
            with self._lock:
                ml = self._loads.get(machine)
            if not ml:
                continue

            if job['status'] == 'running' and ml.busy:
                # Machine is busy — pause this job
                log.info('Throttle: pausing job %s on %s (GPU=%.0f%%, CPU=%.1f)',
                         job['id'], machine, ml.gpu_util, ml.cpu_load_1m)
                self._job_queue.suspend(job['id'])

            elif job['status'] == 'suspended' and not ml.busy:
                # Machine is idle — resume this job
                log.info('Throttle: resuming job %s on %s (GPU=%.0f%%, CPU=%.1f)',
                         job['id'], machine, ml.gpu_util, ml.cpu_load_1m)
                self._job_queue.resume(job['id'])

    def _get_throttled_jobs(self) -> list[dict]:
        """Get running/suspended jobs with throttle enabled."""
        all_jobs = self._job_queue.list_all()
        return [j for j in all_jobs
                if j.get('throttle') and j['status'] in ('running', 'suspended')
                and j.get('machine')]

    # ------------------------------------------------------------------
    # Remote polling
    # ------------------------------------------------------------------

    def _poll_machine(self, machine: str) -> MachineLoad:
        """Poll GPU + CPU load on a machine via SSH."""
        mcfg = get_machine(machine)
        if not mcfg:
            return MachineLoad(machine=machine, error='Machine not found', ts=time.time())

        host = mcfg.get('host', '')
        user = mcfg.get('user', '')
        is_local = mcfg.get('is_local', False)

        ml = MachineLoad(machine=machine, ts=time.time())

        # GPU load (nvidia-smi)
        gpu_cmd = 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
        gpu_out = self._ssh_cmd(host, user, gpu_cmd, is_local)
        if gpu_out:
            try:
                parts = gpu_out.strip().split(',')
                ml.gpu_util = float(parts[0].strip())
                ml.gpu_mem_used = float(parts[1].strip())
                ml.gpu_mem_total = float(parts[2].strip())
            except (ValueError, IndexError):
                pass

        # CPU load (uptime + nproc)
        cpu_cmd = "cat /proc/loadavg 2>/dev/null | cut -d' ' -f1; nproc 2>/dev/null || echo 1"
        cpu_out = self._ssh_cmd(host, user, cpu_cmd, is_local)
        if cpu_out:
            lines = cpu_out.strip().split('\n')
            try:
                ml.cpu_load_1m = float(lines[0].strip())
                ml.cpu_count = int(lines[1].strip()) if len(lines) > 1 else 1
            except (ValueError, IndexError):
                pass

        # Determine if busy
        gpu_busy = ml.gpu_util > self._gpu_threshold
        cpu_busy = ml.cpu_count > 0 and (ml.cpu_load_1m / ml.cpu_count) > self._cpu_threshold
        ml.busy = gpu_busy or cpu_busy

        return ml

    def _ssh_cmd(self, host: str, user: str, cmd: str, is_local: bool) -> str | None:
        """Run a command via SSH (or locally) and return stdout."""
        try:
            if is_local:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=10)
            else:
                ssh_target = f'{user}@{host}' if user else host
                result = subprocess.run(
                    ['ssh', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
                     ssh_target, cmd],
                    capture_output=True, text=True, timeout=10)
            return result.stdout if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, OSError):
            return None

    def _notify_load(self, ml: MachineLoad):
        data = ml.to_dict()
        for cb in self._listeners:
            try:
                cb(data)
            except Exception:
                log.exception('ComputeWatcher listener error')
