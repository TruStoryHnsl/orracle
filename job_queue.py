"""Unified job queue for orracle.

Manages all work across both Studio (generation) and Workshop (training/pipeline):
- Text generation requests
- Image generation jobs (ComfyUI workflows)
- Training jobs (LoRA, QLoRA, SFT)
- Pipeline processing tasks
- Export/deploy tasks

Features:
- Priority queue with categories
- Machine affinity and smart routing
- Automatic failover on machine/service failure
- Parallelization across machines
- Persistent queue (survives server restart)
- Event callbacks for SSE streaming
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from services import ServiceManager

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    WAITING = 'waiting'      # pre-planned, not yet submitted to queue
    PENDING = 'pending'
    ROUTING = 'routing'      # finding a machine
    RUNNING = 'running'
    SUSPENDED = 'suspended'  # paused (SIGSTOP), can resume
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    RETRYING = 'retrying'


class JobCategory(str, Enum):
    GEN_TEXT = 'gen_text'
    GEN_IMAGE = 'gen_image'
    TRAIN = 'train'
    PIPELINE = 'pipeline'
    EXPORT = 'export'


# Service requirements and default priority per category
CATEGORY_CONFIG = {
    JobCategory.GEN_TEXT:  {'services': ['ollama'],  'priority': 1},
    JobCategory.GEN_IMAGE: {'services': ['comfyui'], 'priority': 2},
    JobCategory.TRAIN:     {'services': [],          'priority': 3},
    JobCategory.PIPELINE:  {'services': [],          'priority': 4},
    JobCategory.EXPORT:    {'services': [],          'priority': 5},
}

MAX_RETRIES = 3
MAX_COMPLETED_HISTORY = 200


@dataclass
class Job:
    id: str
    category: JobCategory
    params: dict
    status: JobStatus = JobStatus.PENDING
    priority: int = 5
    machine: str = ''           # assigned machine
    machine_affinity: str = ''  # preferred machine
    submitted: float = 0
    started: float = 0
    finished: float = 0
    progress: float = 0         # 0.0 - 1.0
    progress_msg: str = ''
    error: str = ''
    result: dict = field(default_factory=dict)
    retries: int = 0
    throttle: bool = False       # enable compute load watcher for this job
    remote_pid: int = 0          # PID on remote machine (for suspend/resume)
    session_name: str = ''       # screen session name on remote
    _handle: object = None       # subprocess, thread, etc. (not persisted)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'category': self.category.value,
            'params': self.params,
            'status': self.status.value,
            'priority': self.priority,
            'machine': self.machine,
            'machine_affinity': self.machine_affinity,
            'submitted': self.submitted,
            'started': self.started,
            'finished': self.finished,
            'progress': self.progress,
            'progress_msg': self.progress_msg,
            'error': self.error,
            'result': self.result,
            'retries': self.retries,
            'throttle': self.throttle,
            'remote_pid': self.remote_pid,
            'session_name': self.session_name,
        }

    @staticmethod
    def from_dict(d: dict) -> 'Job':
        return Job(
            id=d['id'],
            category=JobCategory(d['category']),
            params=d.get('params', {}),
            status=JobStatus(d.get('status', 'pending')),
            priority=d.get('priority', 5),
            machine=d.get('machine', ''),
            machine_affinity=d.get('machine_affinity', ''),
            throttle=d.get('throttle', False),
            remote_pid=d.get('remote_pid', 0),
            session_name=d.get('session_name', ''),
            submitted=d.get('submitted', 0),
            started=d.get('started', 0),
            finished=d.get('finished', 0),
            progress=d.get('progress', 0),
            progress_msg=d.get('progress_msg', ''),
            error=d.get('error', ''),
            result=d.get('result', {}),
            retries=d.get('retries', 0),
        )


class JobQueue:
    """Unified job queue with routing, failover, and persistence."""

    def __init__(self, services: 'ServiceManager', config_dir: str):
        self._services = services
        self._config_dir = config_dir
        self._persist_path = os.path.join(config_dir, 'queue.yaml')

        self._pending: list[tuple[int, float, str]] = []  # heapq: (priority, time, job_id)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._listeners: list = []

        self._processor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._load()

    # ------------------------------------------------------------------
    # Submit / Cancel / Retry
    # ------------------------------------------------------------------

    def submit(self, category: str | JobCategory, params: dict,
               machine_affinity: str = '', priority: int = 0) -> str:
        """Submit a job to the queue. Returns job_id."""
        if isinstance(category, str):
            category = JobCategory(category)

        if priority == 0:
            priority = CATEGORY_CONFIG[category]['priority']

        job = Job(
            id=str(uuid.uuid4())[:12],
            category=category,
            params=params,
            priority=priority,
            machine_affinity=machine_affinity,
            submitted=time.time(),
        )

        with self._lock:
            self._jobs[job.id] = job
            heapq.heappush(self._pending, (priority, job.submitted, job.id))
            self._persist()
            self._notify('job_submitted', job)

        log.info('Job submitted: %s (%s)', job.id, category.value)
        return job.id

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            if job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
                return False

            old_status = job.status
            job.status = JobStatus.CANCELLED
            job.finished = time.time()

            # Kill running process if applicable
            if old_status == JobStatus.RUNNING and job._handle:
                self._kill_handle(job)

            self._persist()
            self._notify('job_cancelled', job)

        log.info('Job cancelled: %s', job_id)
        return True

    def retry(self, job_id: str, target_machine: str = '') -> str | None:
        """Retry a failed job, optionally on a different machine.

        Returns new job_id, or None if job not found/not failed.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.FAILED:
                return None

        # Submit a new job with the same params
        affinity = target_machine or job.machine_affinity
        new_id = self.submit(job.category, job.params,
                             machine_affinity=affinity, priority=job.priority)

        with self._lock:
            new_job = self._jobs.get(new_id)
            if new_job:
                new_job.retries = job.retries + 1

        log.info('Job retried: %s -> %s (on %s)', job_id, new_id,
                 target_machine or 'auto')
        return new_id

    # ------------------------------------------------------------------
    # Plan / Start / Suspend / Resume
    # ------------------------------------------------------------------

    def plan_job(self, category: str | JobCategory, params: dict,
                 machine_affinity: str = '', name: str = '',
                 throttle: bool = False) -> str:
        """Pre-plan a job without submitting it. Shows on dashboard as 'waiting'.

        The job sits in WAITING state until start_waiting() is called.
        """
        if isinstance(category, str):
            category = JobCategory(category)

        job = Job(
            id=str(uuid.uuid4())[:12],
            category=category,
            params={**params, 'name': name or params.get('name', '')},
            status=JobStatus.WAITING,
            priority=CATEGORY_CONFIG[category]['priority'],
            machine_affinity=machine_affinity,
            throttle=throttle,
            submitted=time.time(),
        )

        with self._lock:
            self._jobs[job.id] = job
            self._persist()
            self._notify('job_planned', job)

        log.info('Job planned: %s (%s) — waiting for manual start',
                 job.id, category.value)
        return job.id

    def start_waiting(self, job_id: str) -> bool:
        """Start a waiting job — moves it to PENDING in the queue."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.WAITING:
                return False

            job.status = JobStatus.PENDING
            heapq.heappush(
                self._pending, (job.priority, job.submitted, job.id))
            self._persist()
            self._notify('job_submitted', job)

        log.info('Waiting job started: %s', job_id)
        return True

    def suspend(self, job_id: str) -> bool:
        """Suspend a running job (SIGSTOP on remote). Can resume later."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.RUNNING:
                return False

        # Send SIGSTOP to remote process
        if job.remote_pid and job.machine:
            ok = self._remote_signal(job.machine, job.remote_pid, 'STOP')
            if not ok:
                return False

        with self._lock:
            job.status = JobStatus.SUSPENDED
            self._persist()
            self._notify('job_suspended', job)

        log.info('Job suspended: %s (PID %d on %s)',
                 job_id, job.remote_pid, job.machine)
        return True

    def resume(self, job_id: str) -> bool:
        """Resume a suspended job (SIGCONT on remote)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.SUSPENDED:
                return False

        # Send SIGCONT to remote process
        if job.remote_pid and job.machine:
            ok = self._remote_signal(job.machine, job.remote_pid, 'CONT')
            if not ok:
                return False

        with self._lock:
            job.status = JobStatus.RUNNING
            self._persist()
            self._notify('job_resumed', job)

        log.info('Job resumed: %s', job_id)
        return True

    def _remote_signal(self, machine: str, pid: int, signal: str) -> bool:
        """Send a signal to a process on a remote machine via SSH."""
        import subprocess
        from shared import get_machine
        m = get_machine(machine)
        if not m:
            return False
        host = m.get('host', '')
        user = m.get('user', '')
        ssh_target = f'{user}@{host}' if user else host
        try:
            subprocess.run(
                ['ssh', '-o', 'ConnectTimeout=5', ssh_target,
                 f'kill -{signal} {pid}'],
                timeout=10, capture_output=True, check=True,
            )
            return True
        except Exception:
            log.exception('Failed to send %s to PID %d on %s',
                          signal, pid, machine)
            return False

    def list_waiting(self) -> list[dict]:
        """List all waiting (pre-planned) jobs."""
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()
                    if j.status == JobStatus.WAITING]

    def set_throttle(self, job_id: str, enabled: bool) -> bool:
        """Enable or disable compute load throttling for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.throttle = enabled
            self._persist()
            self._notify('job_throttle', job)
        return True

    # ------------------------------------------------------------------
    # Status / Listing
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> dict | None:
        """Get a job's current state."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def update_progress(self, job_id: str, progress: float,
                        message: str = ''):
        """Update a running job's progress (called by dispatchers)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.RUNNING:
                return
            job.progress = min(1.0, max(0.0, progress))
            job.progress_msg = message
            self._notify('job_progress', job)

    def complete(self, job_id: str, result: dict = None):
        """Mark a job as completed (called by dispatchers)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.COMPLETED
            job.finished = time.time()
            job.progress = 1.0
            job.result = result or {}
            self._persist()
            self._notify('job_completed', job)
        log.info('Job completed: %s', job_id)

    def fail(self, job_id: str, error: str, auto_retry: bool = True):
        """Mark a job as failed (called by dispatchers)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.FAILED
            job.finished = time.time()
            job.error = error
            self._persist()
            self._notify('job_failed', job)

        log.warning('Job failed: %s — %s', job_id, error)

        # Auto-retry on a different machine if retries remain
        if auto_retry and job.retries < MAX_RETRIES:
            self.retry(job_id)

    def list_all(self, category: str = '', status: str = '') -> list[dict]:
        """List jobs, optionally filtered."""
        with self._lock:
            jobs = list(self._jobs.values())

        if category:
            jobs = [j for j in jobs if j.category.value == category]
        if status:
            jobs = [j for j in jobs if j.status.value == status]

        # Sort: running first, then pending, then by submitted desc
        status_order = {
            JobStatus.RUNNING: 0, JobStatus.ROUTING: 1,
            JobStatus.PENDING: 2, JobStatus.RETRYING: 3,
            JobStatus.FAILED: 4, JobStatus.COMPLETED: 5,
            JobStatus.CANCELLED: 6,
        }
        jobs.sort(key=lambda j: (status_order.get(j.status, 9), -j.submitted))
        return [j.to_dict() for j in jobs]

    def counts(self) -> dict:
        """Quick status counts."""
        with self._lock:
            c = {}
            for j in self._jobs.values():
                c[j.status.value] = c.get(j.status.value, 0) + 1
            return c

    # ------------------------------------------------------------------
    # Queue processor
    # ------------------------------------------------------------------

    def start_processor(self):
        """Start the background queue processor thread."""
        if self._processor_thread and self._processor_thread.is_alive():
            return
        self._stop_event.clear()
        self._processor_thread = threading.Thread(
            target=self._process_loop, daemon=True)
        self._processor_thread.start()
        log.info('Job queue processor started')

    def stop_processor(self):
        """Stop the queue processor."""
        self._stop_event.set()
        if self._processor_thread:
            self._processor_thread.join(timeout=5)

    def _process_loop(self):
        """Main loop: pick pending jobs and dispatch them."""
        while not self._stop_event.is_set():
            try:
                self._process_next()
            except Exception:
                log.exception('Error in queue processor')
            self._stop_event.wait(2)

    def _process_next(self):
        """Try to dispatch the next pending job."""
        with self._lock:
            # Find the highest-priority pending job
            while self._pending:
                _, _, job_id = self._pending[0]
                job = self._jobs.get(job_id)
                if not job or job.status != JobStatus.PENDING:
                    heapq.heappop(self._pending)
                    continue
                break
            else:
                return  # queue empty

            # Route it
            job.status = JobStatus.ROUTING
            category_cfg = CATEGORY_CONFIG[job.category]
            required = category_cfg['services']

        # Find a machine (outside lock — may do network calls)
        machine = self._route(job, required)
        if not machine:
            # No machine available — leave in queue, try again next cycle
            with self._lock:
                job.status = JobStatus.PENDING
            return

        # Pop from pending and dispatch
        with self._lock:
            heapq.heappop(self._pending)
            job.machine = machine
            job.status = JobStatus.RUNNING
            job.started = time.time()
            self._persist()
            self._notify('job_started', job)

        # Dispatch in a thread so we don't block the processor
        threading.Thread(
            target=self._dispatch, args=(job,), daemon=True).start()

    def _route(self, job: Job, required_services: list[str]) -> str | None:
        """Find the best machine for a job."""
        # Prefer affinity machine if healthy
        if job.machine_affinity:
            if not required_services or \
               self._services.is_healthy(job.machine_affinity, required_services):
                return job.machine_affinity

        if not required_services:
            # CPU-bound jobs run locally
            return 'local'

        # Find machines with required services online
        online = self._services.find_online(required_services[0])
        if not online:
            return None

        # Pick the machine with fewest active jobs
        with self._lock:
            load = {}
            for j in self._jobs.values():
                if j.status == JobStatus.RUNNING and j.machine:
                    load[j.machine] = load.get(j.machine, 0) + 1

        candidates = [(s['machine'], load.get(s['machine'], 0)) for s in online]
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0] if candidates else None

    def _dispatch(self, job: Job):
        """Execute a job. This runs in a worker thread."""
        try:
            if job.category == JobCategory.TRAIN:
                self._dispatch_training(job)
            elif job.category == JobCategory.GEN_IMAGE:
                self._dispatch_image_gen(job)
            elif job.category == JobCategory.PIPELINE:
                self._dispatch_pipeline(job)
            elif job.category == JobCategory.EXPORT:
                self._dispatch_export(job)
            else:
                self.fail(job.id, f'Unknown category: {job.category}',
                          auto_retry=False)
        except Exception as e:
            self.fail(job.id, str(e))

    # ------------------------------------------------------------------
    # Category dispatchers (integrate with existing backend modules)
    # ------------------------------------------------------------------

    def _dispatch_training(self, job: Job):
        """Dispatch a training job using existing training infrastructure."""
        from training import jobs as train_jobs, remote

        params = job.params
        machine = job.machine

        if machine == 'local':
            result = train_jobs.start_job(params)
            if result.get('error'):
                self.fail(job.id, result['error'], auto_retry=False)
                return
            # Monitor the local training job
            inner_id = result.get('job_id')
            while True:
                inner = train_jobs.get_job(inner_id) if inner_id else None
                if not inner:
                    break
                if inner.get('status') in ('completed', 'failed', 'cancelled'):
                    break
                self._stop_event.wait(10)

            if inner and inner.get('status') == 'completed':
                self.complete(job.id, inner)
            else:
                self.fail(job.id, inner.get('error', 'Training ended'),
                          auto_retry=False)
        else:
            # Remote training via SSH
            from shared import get_machine
            m = get_machine(machine)
            if not m:
                self.fail(job.id, f'Machine {machine} not found',
                          auto_retry=False)
                return
            result = remote.start_remote_training(
                m.get('host', ''),
                m.get('project_path', '~/projects/orracle'),
                params.get('config_file', ''),
                venv=params.get('venv', 'venv_mlx'),
                session_name=params.get('session_name', 'orrvert'),
            )
            if result.get('error'):
                self.fail(job.id, result['error'])
            else:
                # Training is fire-and-forget on remote — mark as running
                # The monitor system tracks progress separately
                self.update_progress(job.id, 0.01, 'Training started on remote')

    def _dispatch_image_gen(self, job: Job):
        """Dispatch an image generation job via ComfyUI with WebSocket progress."""
        from training import comfyui

        # Prefer URL from job params (set by API), fall back to service manager
        url = job.params.get('comfyui_url') or \
            self._services.get_endpoint(job.machine, 'comfyui')

        # Auto-start ComfyUI if unavailable
        if not url:
            url = self._try_autostart_comfyui(job.machine)
        if not url:
            self.fail(job.id, 'No ComfyUI endpoint available')
            return

        workflow = job.params.get('workflow', {})
        self.update_progress(job.id, 0.05, 'Submitting to ComfyUI')

        # Progress callback — relays ComfyUI WebSocket events to job queue SSE
        current_node = [None]
        total_nodes = [max(1, len(workflow))]

        def on_progress(event_type, data):
            if event_type == 'queued':
                self.update_progress(job.id, 0.1, 'Queued in ComfyUI')
            elif event_type == 'execution_start':
                self.update_progress(job.id, 0.15, 'Generating...')
            elif event_type == 'executing':
                node = data.get('node')
                if node:
                    current_node[0] = node
                    # Estimate progress from node execution order
                    node_class = workflow.get(node, {}).get('class_type', '')
                    pct = 0.15 + 0.7 * (list(workflow.keys()).index(node) + 1) / total_nodes[0] \
                        if node in workflow else 0.5
                    self.update_progress(job.id, min(0.85, pct),
                                         f'Running {node_class}' if node_class else 'Processing...')
            elif event_type == 'progress':
                # Step-level progress within a node (e.g., KSampler steps)
                value = data.get('value', 0)
                max_val = data.get('max', 1)
                if max_val > 0:
                    step_pct = value / max_val
                    self.update_progress(
                        job.id, 0.2 + 0.65 * step_pct,
                        f'Step {value}/{max_val}')
            elif event_type == 'executed':
                self.update_progress(job.id, 0.9, 'Saving output...')

        # Stream via WebSocket (falls back to polling if unavailable)
        history = comfyui.stream_prompt(url, workflow, timeout=600,
                                        on_progress=on_progress)
        if history:
            prompt_id = history.get('prompt_id', '')
            # Extract prompt_id from the outputs if not at top level
            if not prompt_id:
                for node_out in history.get('outputs', {}).values():
                    for img in node_out.get('images', []):
                        prompt_id = img.get('prompt_id', prompt_id)
                        break
            self.complete(job.id, {'prompt_id': prompt_id, 'output': history})
        else:
            self.fail(job.id, 'ComfyUI prompt timed out or failed')

    def _try_autostart_comfyui(self, machine: str) -> str | None:
        """Attempt to auto-start ComfyUI on the target machine.

        Returns the endpoint URL if successful, None otherwise.
        """
        # Find a ComfyUI service on this machine
        services = self._services.get_by_type('comfyui')
        target = None
        for svc in services:
            if svc['machine'] == machine:
                target = svc
                break

        if not target:
            # No ComfyUI configured on this machine at all
            return None

        if target['status'] == 'online':
            return target['url']

        if not target.get('start_cmd'):
            return None

        log.info('Auto-starting ComfyUI on %s', machine)
        result = self._services.start_service(machine, target['name'])
        if not result.get('ok'):
            log.warning('Failed to auto-start ComfyUI on %s: %s',
                        machine, result.get('error'))
            return None

        # Wait for it to come online (start_service already spawns a wait thread,
        # but we need to block here until it's ready)
        deadline = time.time() + 120
        while time.time() < deadline:
            status = self._services.check_health(machine, target['name'])
            if status.value == 'online':
                return target['url']
            self._stop_event.wait(3)

        log.warning('ComfyUI auto-start timed out on %s', machine)
        return None

    def _dispatch_pipeline(self, job: Job):
        """Dispatch a data pipeline job."""
        self.update_progress(job.id, 0.1, 'Pipeline starting')
        # Pipeline dispatch integrates with executor/remote.py
        # For now, mark as completed — full integration in Phase 5
        self.complete(job.id, {'message': 'Pipeline dispatch not yet wired'})

    def _dispatch_export(self, job: Job):
        """Dispatch an export/deploy job."""
        self.update_progress(job.id, 0.1, 'Export starting')
        # Export dispatch integrates with training/export_mgr.py
        # For now, mark as completed — full integration in Phase 5
        self.complete(job.id, {'message': 'Export dispatch not yet wired'})

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self):
        """Save queue state to disk (called under lock)."""
        data = {
            'jobs': {},
        }
        # Only persist non-running jobs and recent completed
        completed = []
        for jid, job in self._jobs.items():
            d = job.to_dict()
            if job.status == JobStatus.COMPLETED:
                completed.append((job.finished, jid, d))
            else:
                data['jobs'][jid] = d

        # Keep only recent completed
        completed.sort(reverse=True)
        for _, jid, d in completed[:MAX_COMPLETED_HISTORY]:
            data['jobs'][jid] = d

        try:
            os.makedirs(self._config_dir, exist_ok=True)
            with open(self._persist_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except Exception:
            log.exception('Failed to persist queue')

    def _load(self):
        """Load persisted queue state."""
        try:
            with open(self._persist_path) as f:
                data = yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError):
            return

        for jid, jdata in data.get('jobs', {}).items():
            try:
                job = Job.from_dict(jdata)
                # Running jobs from a previous session are now orphaned
                if job.status in (JobStatus.RUNNING, JobStatus.ROUTING):
                    job.status = JobStatus.FAILED
                    job.error = 'Orphaned after server restart'
                self._jobs[jid] = job

                # Re-queue pending jobs
                if job.status == JobStatus.PENDING:
                    heapq.heappush(
                        self._pending,
                        (job.priority, job.submitted, job.id),
                    )
            except Exception:
                log.exception('Failed to restore job %s', jid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kill_handle(self, job: Job):
        """Kill a running job's process handle."""
        handle = job._handle
        if handle is None:
            return
        try:
            if hasattr(handle, 'terminate'):
                handle.terminate()
            elif hasattr(handle, 'cancel'):
                handle.cancel()
        except Exception:
            log.exception('Failed to kill job handle for %s', job.id)

    # ------------------------------------------------------------------
    # Event listeners
    # ------------------------------------------------------------------

    def add_listener(self, callback):
        """Register a callback for job events.

        Callback signature: callback(event_type: str, job_dict: dict)
        """
        self._listeners.append(callback)

    def remove_listener(self, callback):
        self._listeners = [cb for cb in self._listeners if cb is not callback]

    def _notify(self, event: str, job: Job):
        """Notify listeners of a job event."""
        data = job.to_dict()
        for cb in self._listeners:
            try:
                cb(event, data)
            except Exception:
                log.exception('Error in job queue listener')
