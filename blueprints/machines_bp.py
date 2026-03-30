"""Machines blueprint — hardware detection, remote management, WoL, download queue."""

from __future__ import annotations

import json
import time

from flask import (Blueprint, Response, jsonify, render_template, request,
                   stream_with_context)

from training import hardware, log_parser, remote
from training import export_mgr as export
from shared import (load_machines as _load_machines, save_machines as _save_machines_shared,
                    get_local_hardware as _get_local_hardware, refresh_hardware, CONFIG_DIR)

machines_bp = Blueprint('machines', __name__, url_prefix='/machines')


def _save_machines(machines: dict):
    _save_machines_shared(machines)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@machines_bp.route('/')
def index():
    """Machines overview page."""
    hw = _get_local_hardware()
    machines = _load_machines()
    monitors = remote.list_remote_monitors()
    return render_template('machines/machines.html', hw=hw, machines=machines,
                           remote_monitors=monitors)


# ---------------------------------------------------------------------------
# Hardware API
# ---------------------------------------------------------------------------

@machines_bp.route('/api/hardware')
def api_hardware():
    """Get local hardware info."""
    hw = _get_local_hardware()
    return jsonify(hw)


@machines_bp.route('/api/hardware/refresh', methods=['POST'])
def api_hardware_refresh():
    """Force-refresh local hardware cache."""
    with _hw_lock:
        _hw_cache['data'] = None
        _hw_cache['ts'] = 0
    hw = _get_local_hardware()
    return jsonify(hw)


# ---------------------------------------------------------------------------
# Machine CRUD
# ---------------------------------------------------------------------------

@machines_bp.route('/api/list')
def api_list():
    """List all configured machines."""
    return jsonify(_load_machines())


@machines_bp.route('/api/add', methods=['POST'])
def api_add():
    """Add a new machine."""
    data = request.json or {}
    name = data.get('name', '').strip()
    hostname = data.get('hostname', '').strip()
    if not name or not hostname:
        return jsonify({'error': 'name and hostname required'}), 400

    machines = _load_machines()
    machines[name] = {
        'hostname': hostname,
        'ssh_user': data.get('ssh_user', ''),
        'niftytune_path': data.get('niftytune_path', '~/niftytune'),
        'mac_address': data.get('mac_address', ''),
    }
    _save_machines(machines)
    return jsonify({'ok': True})


@machines_bp.route('/api/<name>/services')
def api_services(name):
    """Probe running services on a machine."""
    machines = _load_machines()
    m = machines.get(name)
    if not m:
        return jsonify({'error': 'Machine not found'}), 404

    hostname = m.get('hostname', '')
    if m.get('is_local'):
        hostname = 'localhost'
    services_cfg = m.get('services', {})
    results = remote.probe_services(hostname,
                                    services_cfg if services_cfg else None)
    return jsonify(results)


# ---------------------------------------------------------------------------
# Remote connection / hardware
# ---------------------------------------------------------------------------

@machines_bp.route('/api/test/<name>', methods=['POST'])
def api_test(name):
    """Test SSH connection to a remote machine."""
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    result = remote.test_connection(m['hostname'])
    return jsonify(result)


@machines_bp.route('/api/hardware/<name>', methods=['POST'])
def api_remote_hardware(name):
    """Detect hardware on a remote machine (via SSH)."""
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    hw = remote.detect_remote_hardware(m['hostname'])
    # Cache in machines config
    machines[name]['_hw'] = hw
    machines[name]['_hw_ts'] = time.strftime('%Y-%m-%d %H:%M:%S')
    _save_machines(machines)
    return jsonify(hw)


# ---------------------------------------------------------------------------
# Remote training detection / logs
# ---------------------------------------------------------------------------

@machines_bp.route('/api/training/<name>')
def api_remote_training(name):
    """Detect running training processes on a remote machine."""
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    detected = remote.detect_remote_training(
        m['hostname'], m.get('niftytune_path', '~/niftytune'))
    return jsonify(detected)


@machines_bp.route('/api/logs/<name>')
def api_remote_logs(name):
    """Find training log files on a remote machine."""
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    logs = remote.find_remote_logs(
        m['hostname'], m.get('niftytune_path', '~/niftytune'))
    return jsonify(logs)


@machines_bp.route('/api/log/<name>', methods=['POST'])
def api_read_log(name):
    """Read and parse a remote training log."""
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    data = request.json or {}
    log_path = data.get('log_path', '').strip()
    tail = int(data.get('tail', 100))
    if not log_path:
        return jsonify({'error': 'log_path required'}), 400
    lines = remote.read_remote_log(m['hostname'], log_path, tail)
    # Parse metrics from lines
    metrics = []
    for line in lines:
        metric = log_parser.parse_line(line)
        if metric:
            metrics.append(metric)
    return jsonify({'lines': lines, 'metrics': metrics})


# ---------------------------------------------------------------------------
# Remote monitor (SSE streaming for remote training)
# ---------------------------------------------------------------------------

@machines_bp.route('/api/monitor/start', methods=['POST'])
def api_monitor_start():
    """Start a remote training log monitor."""
    data = request.json or {}
    name = data.get('machine', '').strip()
    log_path = data.get('log_path', '').strip()
    total_iters = int(data.get('total_iters', 0))
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    if not log_path:
        return jsonify({'error': 'log_path required'}), 400
    monitor_id = remote.start_remote_monitor(
        m['hostname'], log_path, total_iters)
    if not monitor_id:
        return jsonify({'error': 'failed to start monitor'}), 500
    return jsonify({'monitor_id': monitor_id})


@machines_bp.route('/api/monitor/stream/<monitor_id>')
def api_monitor_stream(monitor_id):
    """SSE endpoint for remote training log streaming."""
    def generate():
        sent_lines = 0
        sent_metrics = 0

        waited = 0
        monitor = None
        while waited < 10:
            monitor = remote.get_remote_monitor(monitor_id)
            if monitor:
                break
            time.sleep(1)
            waited += 1

        if not monitor:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Monitor not found'})}\n\n"
            return

        while True:
            with remote._remote_lock:
                lines = monitor['output_lines']
                metrics = monitor['metrics']
                done = monitor['done']
                new_lines = lines[sent_lines:]
                new_metrics = metrics[sent_metrics:]
                sent_lines = len(lines)
                sent_metrics = len(metrics)

            for line in new_lines:
                yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"

            for m in new_metrics:
                yield f"data: {json.dumps({'type': 'metric', **m})}\n\n"

            if done:
                yield f"data: {json.dumps({'type': 'done', 'exit_code': monitor.get('exit_code', 0)})}\n\n"
                break

            if new_metrics:
                total = monitor.get('total_iters', 0)
                eta = log_parser.estimate_eta(metrics, total)
                if eta is not None:
                    yield f"data: {json.dumps({'type': 'eta', 'seconds': eta})}\n\n"

            time.sleep(0.5)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@machines_bp.route('/api/monitor/stop/<monitor_id>', methods=['POST'])
def api_monitor_stop(monitor_id):
    """Stop a remote training monitor."""
    ok = remote.stop_remote_monitor(monitor_id)
    return jsonify({'ok': ok})


@machines_bp.route('/api/monitors')
def api_monitors():
    """List all active remote monitors."""
    return jsonify(remote.list_remote_monitors())


# ---------------------------------------------------------------------------
# Remote training control
# ---------------------------------------------------------------------------

@machines_bp.route('/api/train/start', methods=['POST'])
def api_train_start():
    """Start training on a remote machine."""
    data = request.json or {}
    name = data.get('machine', '').strip()
    config_file = data.get('config_file', '').strip()
    session_name = data.get('session_name', 'orrvert')
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    if not config_file:
        return jsonify({'error': 'config_file required'}), 400
    result = remote.start_remote_training(
        m['hostname'],
        m.get('niftytune_path', '~/niftytune'),
        config_file,
        venv=data.get('venv', 'venv_mlx'),
        session_name=session_name,
    )
    return jsonify(result)


@machines_bp.route('/api/train/stop', methods=['POST'])
def api_train_stop():
    """Stop training on a remote machine."""
    data = request.json or {}
    name = data.get('machine', '').strip()
    session_name = data.get('session_name', 'orrvert')
    pid = data.get('pid')
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    result = remote.stop_remote_training(
        m['hostname'], session_name=session_name, pid=pid)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Wake-on-LAN
# ---------------------------------------------------------------------------

@machines_bp.route('/api/wol/<name>', methods=['POST'])
def api_wol(name):
    """Send Wake-on-LAN packet to a machine."""
    machines = _load_machines()
    m = machines.get(name)
    if not m:
        return jsonify({'error': 'machine not found'}), 404

    mac = m.get('mac_address')
    if not mac:
        hostname = m.get('hostname', '')
        if hostname:
            mac = remote.get_mac_address(hostname)
            if mac:
                machines[name]['mac_address'] = mac
                _save_machines(machines)
        if not mac:
            return jsonify({'error': 'No MAC address configured. '
                            'Add mac_address to machine config.'}), 400

    result = remote.send_wol(mac)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Job queue (for offline machines)
# ---------------------------------------------------------------------------

@machines_bp.route('/api/queue', methods=['POST'])
def api_queue_job():
    """Queue a job for an offline machine."""
    data = request.json or {}
    machine = data.get('machine', '').strip()
    action = data.get('action', '').strip()
    params = data.get('params', {})
    if not machine or not action:
        return jsonify({'error': 'machine and action required'}), 400

    entry_id = remote.queue_job(machine, action, params)
    return jsonify({'id': entry_id, 'status': 'queued'})


@machines_bp.route('/api/queue')
def api_queue_list():
    """List all queued jobs."""
    return jsonify(remote.list_queue())


@machines_bp.route('/api/queue/<entry_id>', methods=['DELETE'])
def api_queue_cancel(entry_id):
    """Cancel a queued job."""
    ok = remote.cancel_queued(entry_id)
    return jsonify({'ok': ok})


# ---------------------------------------------------------------------------
# Remote adapters & downloads
# ---------------------------------------------------------------------------

@machines_bp.route('/api/<name>/adapters')
def api_remote_adapters(name):
    """List LoRA adapters on a remote machine."""
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    adapters = remote.list_remote_adapters(
        m['hostname'], m.get('niftytune_path', '~/niftytune'))
    return jsonify(adapters)


@machines_bp.route('/api/download-adapter', methods=['POST'])
def api_download_adapter():
    """Download an adapter from a remote machine to local scan path."""
    data = request.json or {}
    name = data.get('machine', '').strip()
    remote_path = data.get('remote_path', '').strip()
    if not name or not remote_path:
        return jsonify({'error': 'machine and remote_path required'}), 400

    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404

    local_dir = str(export.NIFTYTUNE_DIR / 'adapters')
    task_id = remote.download_adapter(m['hostname'], remote_path, local_dir)
    return jsonify({'task_id': task_id, 'status': 'started',
                    'local_dir': local_dir})


@machines_bp.route('/api/download/<task_id>')
def api_download_status(task_id):
    """Check status of an adapter download."""
    task = remote.get_download_task(task_id)
    if not task:
        return jsonify({'error': 'not found'}), 404
    return jsonify(task)


# ---------------------------------------------------------------------------
# Remote checkpoints
# ---------------------------------------------------------------------------

@machines_bp.route('/api/checkpoints/<name>')
def api_remote_checkpoints(name):
    """List training checkpoints on a remote machine."""
    machines = _load_machines()
    m = machines.get(name)
    if not m or not m.get('hostname'):
        return jsonify({'error': 'machine not found'}), 404
    data = request.args
    adapter_dir = data.get('dir',
                           m.get('niftytune_path', '~/niftytune') + '/adapters_hq')
    return jsonify(remote.list_remote_checkpoints(m['hostname'], adapter_dir))
