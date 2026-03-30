"""Dashboard blueprint — unified command center.

Shows real-time status of all machines, services, and jobs.
Provides SSE streaming for live updates and quick actions
for service control and job management.
"""

from __future__ import annotations

import json
import queue
import threading
import time

from flask import (Blueprint, Response, current_app, jsonify,
                   render_template, request, stream_with_context)

from shared import get_local_hardware, load_machines
from training import jobs, hardware, remote
from training import export_mgr as export

dashboard_bp = Blueprint('dashboard', __name__)

# SSE client queues — each connected dashboard gets one
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _broadcast(event_type: str, data: dict):
    """Push an event to all connected SSE clients."""
    msg = json.dumps({'type': event_type, **data})
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


def _setup_listeners(app):
    """Wire service, job queue, and compute watcher listeners to SSE broadcast."""
    svc_mgr = app.config.get('service_manager')
    job_queue = app.config.get('job_queue')
    watcher = app.config.get('compute_watcher')

    if svc_mgr:
        svc_mgr.add_listener(
            lambda data: _broadcast('service_change', data))

    if job_queue:
        job_queue.add_listener(
            lambda event, data: _broadcast(event, {'job': data}))

    if watcher:
        watcher.add_listener(
            lambda data: _broadcast('compute_load', {'load': data}))


_listeners_setup = False


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------

@dashboard_bp.route('/')
def index():
    """Unified dashboard — command center."""
    global _listeners_setup
    if not _listeners_setup:
        _setup_listeners(current_app._get_current_object())
        _listeners_setup = True

    hw = get_local_hardware()
    machines = load_machines()

    # Service states from service manager
    svc_mgr = current_app.config.get('service_manager')
    services = svc_mgr.get_all() if svc_mgr else []

    # Job queue
    job_queue = current_app.config.get('job_queue')
    queue_jobs = job_queue.list_all() if job_queue else []
    queue_counts = job_queue.counts() if job_queue else {}

    active_jobs = [j for j in queue_jobs
                   if j['status'] in ('running', 'pending', 'routing', 'suspended')]
    waiting_jobs = [j for j in queue_jobs
                    if j['status'] == 'waiting']
    recent_jobs = [j for j in queue_jobs
                   if j['status'] in ('completed', 'failed', 'cancelled')][:10]

    # Also include legacy training jobs
    legacy_jobs = jobs.list_jobs()
    legacy_active = []
    for job in legacy_jobs:
        if job['status'] == 'running':
            jdata = jobs.get_active_job(job['id'])
            if jdata:
                recent_train = [m for m in jdata['metrics']
                                if m.get('type') == 'train']
                job['_live'] = {
                    'current_iter': (recent_train[-1]['iter']
                                     if recent_train else 0),
                    'current_loss': (recent_train[-1]['train_loss']
                                     if recent_train else None),
                    'best_val_loss': jdata['best_val_loss'],
                    'total_iters': jdata['total_iters'],
                }
            elif job.get('monitor_id'):
                monitor = remote.get_remote_monitor(job['monitor_id'])
                if monitor:
                    with remote._remote_lock:
                        recent_train = [m for m in monitor['metrics']
                                        if m.get('type') == 'train']
                    job['_live'] = {
                        'current_iter': (recent_train[-1]['iter']
                                         if recent_train else 0),
                        'current_loss': (recent_train[-1]['train_loss']
                                         if recent_train else None),
                        'best_val_loss': monitor.get('best_val_loss'),
                        'total_iters': monitor.get('total_iters',
                                                   job.get('total_iters', 0)),
                    }
            legacy_active.append(job)
    legacy_recent = [j for j in legacy_jobs
                     if j['status'] in ('completed', 'failed', 'cancelled')][:5]

    # Group services by machine for the machine bar
    machines_status = {}
    for svc in services:
        mname = svc['machine']
        if mname not in machines_status:
            mcfg = machines.get(mname, {})
            machines_status[mname] = {
                'name': mname,
                'description': mcfg.get('description', ''),
                'services': [],
                'any_online': False,
                'all_online': True,
            }
        machines_status[mname]['services'].append(svc)
        if svc['status'] == 'online':
            machines_status[mname]['any_online'] = True
        else:
            machines_status[mname]['all_online'] = False

    # Ollama models
    try:
        ollama_models = export.list_ollama_models()
    except Exception:
        ollama_models = []

    # Compute load snapshots (from watcher)
    watcher = current_app.config.get('compute_watcher')
    compute_loads = {}
    if watcher:
        for load in watcher.get_all_loads():
            compute_loads[load['machine']] = load

    return render_template('dashboard_new.html',
                           hw=hw,
                           machines_status=machines_status,
                           services=services,
                           active_jobs=active_jobs,
                           waiting_jobs=waiting_jobs,
                           recent_jobs=recent_jobs,
                           legacy_active=legacy_active,
                           legacy_recent=legacy_recent,
                           queue_counts=queue_counts,
                           ollama_models=ollama_models,
                           compute_loads=compute_loads,
                           gpu_summary=hardware.format_gpu_summary(hw),
                           ram_summary=hardware.format_ram_summary(hw))


# ---------------------------------------------------------------------------
# Dashboard API
# ---------------------------------------------------------------------------

@dashboard_bp.route('/api/dashboard/status')
def api_status():
    """Full status snapshot for dashboard widgets and nav dots."""
    svc_mgr = current_app.config.get('service_manager')
    job_queue = current_app.config.get('job_queue')

    services = svc_mgr.get_all() if svc_mgr else []
    queue_counts = job_queue.counts() if job_queue else {}

    machines = load_machines()
    monitors = remote.list_remote_monitors()

    return jsonify({
        'services': services,
        'queue': queue_counts,
        'machines': len(machines),
        'active_monitors': len([m for m in monitors if not m.get('done')]),
    })


@dashboard_bp.route('/api/dashboard/stream')
def api_stream():
    """SSE endpoint for real-time dashboard updates."""
    client_queue = queue.Queue(maxsize=100)

    with _sse_lock:
        _sse_clients.append(client_queue)

    def generate():
        try:
            # Send initial heartbeat
            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

            while True:
                try:
                    msg = client_queue.get(timeout=15)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            with _sse_lock:
                if client_queue in _sse_clients:
                    _sse_clients.remove(client_queue)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@dashboard_bp.route('/api/dashboard/action', methods=['POST'])
def api_action():
    """Handle quick actions from the dashboard."""
    data = request.json or {}
    action = data.get('action', '')

    svc_mgr = current_app.config.get('service_manager')
    job_queue = current_app.config.get('job_queue')

    if action == 'start_service':
        machine = data.get('machine', '')
        service = data.get('service', '')
        if not svc_mgr:
            return jsonify({'ok': False, 'error': 'Service manager not available'})
        result = svc_mgr.start_service(machine, service)
        return jsonify(result)

    elif action == 'stop_service':
        machine = data.get('machine', '')
        service = data.get('service', '')
        if not svc_mgr:
            return jsonify({'ok': False, 'error': 'Service manager not available'})
        result = svc_mgr.stop_service(machine, service)
        return jsonify(result)

    elif action == 'cancel_job':
        job_id = data.get('job_id', '')
        if job_queue:
            ok = job_queue.cancel(job_id)
            return jsonify({'ok': ok})
        return jsonify({'ok': False, 'error': 'Queue not available'})

    elif action == 'retry_job':
        job_id = data.get('job_id', '')
        target = data.get('target_machine', '')
        if job_queue:
            new_id = job_queue.retry(job_id, target)
            return jsonify({'ok': bool(new_id), 'new_id': new_id})
        return jsonify({'ok': False, 'error': 'Queue not available'})

    elif action == 'start_waiting':
        job_id = data.get('job_id', '')
        if job_queue:
            ok = job_queue.start_waiting(job_id)
            return jsonify({'ok': ok})
        return jsonify({'ok': False, 'error': 'Queue not available'})

    elif action == 'suspend_job':
        job_id = data.get('job_id', '')
        if job_queue:
            ok = job_queue.suspend(job_id)
            return jsonify({'ok': ok})
        return jsonify({'ok': False, 'error': 'Queue not available'})

    elif action == 'resume_job':
        job_id = data.get('job_id', '')
        if job_queue:
            ok = job_queue.resume(job_id)
            return jsonify({'ok': ok})
        return jsonify({'ok': False, 'error': 'Queue not available'})

    elif action == 'toggle_throttle':
        job_id = data.get('job_id', '')
        if job_queue:
            ok = job_queue.set_throttle(job_id, data.get('throttle', False))
            return jsonify({'ok': ok})
        return jsonify({'ok': False, 'error': 'Queue not available'})

    return jsonify({'ok': False, 'error': f'Unknown action: {action}'})


# ---------------------------------------------------------------------------
# Service and queue API shortcuts (used by nav dots and dashboard JS)
# ---------------------------------------------------------------------------

@dashboard_bp.route('/api/services')
def api_services():
    """List all service states."""
    svc_mgr = current_app.config.get('service_manager')
    return jsonify(svc_mgr.get_all() if svc_mgr else [])


@dashboard_bp.route('/api/services/<machine>/<service>/start', methods=['POST'])
def api_service_start(machine, service):
    """Start a specific service."""
    svc_mgr = current_app.config.get('service_manager')
    if not svc_mgr:
        return jsonify({'ok': False, 'error': 'Not available'})
    return jsonify(svc_mgr.start_service(machine, service))


@dashboard_bp.route('/api/services/<machine>/<service>/stop', methods=['POST'])
def api_service_stop(machine, service):
    """Stop a specific service."""
    svc_mgr = current_app.config.get('service_manager')
    if not svc_mgr:
        return jsonify({'ok': False, 'error': 'Not available'})
    return jsonify(svc_mgr.stop_service(machine, service))


@dashboard_bp.route('/api/queue/list')
def api_queue_list():
    """List all queued jobs."""
    job_queue = current_app.config.get('job_queue')
    category = request.args.get('category', '')
    status = request.args.get('status', '')
    return jsonify(job_queue.list_all(category, status) if job_queue else [])


@dashboard_bp.route('/api/queue/counts')
def api_queue_counts():
    """Get job status counts."""
    job_queue = current_app.config.get('job_queue')
    return jsonify(job_queue.counts() if job_queue else {})


@dashboard_bp.route('/api/queue/<job_id>/cancel', methods=['POST'])
def api_queue_cancel(job_id):
    """Cancel a job."""
    job_queue = current_app.config.get('job_queue')
    if not job_queue:
        return jsonify({'ok': False})
    return jsonify({'ok': job_queue.cancel(job_id)})


@dashboard_bp.route('/api/queue/<job_id>/retry', methods=['POST'])
def api_queue_retry(job_id):
    """Retry a failed job."""
    job_queue = current_app.config.get('job_queue')
    if not job_queue:
        return jsonify({'ok': False})
    data = request.json or {}
    new_id = job_queue.retry(job_id, data.get('target_machine', ''))
    return jsonify({'ok': bool(new_id), 'new_id': new_id})


@dashboard_bp.route('/api/queue/plan', methods=['POST'])
def api_queue_plan():
    """Pre-plan a job (WAITING state) for later manual start."""
    job_queue = current_app.config.get('job_queue')
    if not job_queue:
        return jsonify({'ok': False, 'error': 'Queue not available'})

    data = request.json or {}
    category = data.get('category', 'train')
    params = data.get('params', {})
    machine = data.get('machine_affinity', '')
    name = data.get('name', '')
    throttle = data.get('throttle', False)

    job_id = job_queue.plan_job(category, params,
                                machine_affinity=machine,
                                name=name, throttle=throttle)
    return jsonify({'ok': True, 'job_id': job_id})


@dashboard_bp.route('/api/queue/waiting')
def api_queue_waiting():
    """List all waiting (pre-planned) jobs."""
    job_queue = current_app.config.get('job_queue')
    return jsonify(job_queue.list_waiting() if job_queue else [])


@dashboard_bp.route('/api/queue/<job_id>/start', methods=['POST'])
def api_queue_start(job_id):
    """Start a waiting job."""
    job_queue = current_app.config.get('job_queue')
    if not job_queue:
        return jsonify({'ok': False})
    return jsonify({'ok': job_queue.start_waiting(job_id)})


@dashboard_bp.route('/api/queue/<job_id>/suspend', methods=['POST'])
def api_queue_suspend(job_id):
    """Suspend a running job."""
    job_queue = current_app.config.get('job_queue')
    if not job_queue:
        return jsonify({'ok': False})
    return jsonify({'ok': job_queue.suspend(job_id)})


@dashboard_bp.route('/api/queue/<job_id>/resume', methods=['POST'])
def api_queue_resume(job_id):
    """Resume a suspended job."""
    job_queue = current_app.config.get('job_queue')
    if not job_queue:
        return jsonify({'ok': False})
    return jsonify({'ok': job_queue.resume(job_id)})


@dashboard_bp.route('/api/queue/<job_id>/throttle', methods=['POST'])
def api_queue_throttle(job_id):
    """Toggle compute load throttling for a job."""
    job_queue = current_app.config.get('job_queue')
    if not job_queue:
        return jsonify({'ok': False})
    data = request.json or {}
    return jsonify({'ok': job_queue.set_throttle(job_id, data.get('throttle', False))})


@dashboard_bp.route('/api/compute/loads')
def api_compute_loads():
    """Get current compute load for all monitored machines."""
    watcher = current_app.config.get('compute_watcher')
    if not watcher:
        return jsonify([])
    return jsonify(watcher.get_all_loads())
