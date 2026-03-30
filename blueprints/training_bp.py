"""Training blueprint — train page, job lifecycle, SSE streaming, metrics."""

from __future__ import annotations

import json
import time

from flask import (Blueprint, Response, jsonify, redirect, render_template,
                   request, stream_with_context, url_for)

from training import jobs, log_parser, remote
from shared import load_machines as _load_machines

training_bp = Blueprint('training', __name__, url_prefix='/workshop/train')


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@training_bp.route('/')
def index():
    """Training launch page."""
    from training import hardware
    hw = hardware.detect_hardware()
    machines = _load_machines()
    presets = jobs.MODEL_PRESETS
    return render_template('training/train.html', hw=hw, machines=machines,
                           presets=presets)


@training_bp.route('/monitor/<job_id>')
def monitor_page(job_id):
    """Live monitoring page for a training job."""
    job = jobs.get_job(job_id)
    if not job:
        return redirect(url_for('dashboard.index'))
    monitor_id = job.get('monitor_id', '')
    is_remote = job.get('machine', 'local') != 'local'
    return render_template('training/monitor.html', job=job, job_id=job_id,
                           monitor_id=monitor_id, is_remote=is_remote)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@training_bp.route('/api/start', methods=['POST'])
def api_start():
    """Start a training job (local or remote)."""
    data = request.json or {}

    if not data.get('model') and not data.get('model_preset'):
        return jsonify({'error': 'model required'}), 400

    # Resolve preset
    preset = data.get('model_preset')
    if preset and preset in jobs.MODEL_PRESETS:
        data['model'] = jobs.MODEL_PRESETS[preset]['model']

    machine = data.get('machine', 'local')

    # Remote machine — use SSH + screen
    if machine != 'local':
        return _start_remote_job(data, machine)

    # Local machine — use subprocess
    try:
        job_id = jobs.start_job(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'job_id': job_id, 'status': 'started'})


def _start_remote_job(data: dict, machine_name: str):
    """Start a training job on a remote machine."""
    machines = _load_machines()
    m = machines.get(machine_name)
    if not m or not m.get('hostname'):
        return jsonify({'error': f'Machine {machine_name} not found'}), 404

    hostname = m['hostname']
    niftytune_dir = m.get('niftytune_path', '~/niftytune')

    # Generate config locally
    job_id = data.get('job_id') or str(int(time.time()))
    data['job_id'] = job_id
    data['work_dir'] = niftytune_dir

    try:
        config_path = jobs.generate_mlx_config(data)
    except Exception as e:
        return jsonify({'error': f'Config generation failed: {e}'}), 500

    # Upload config to remote machine
    remote_config = f'{niftytune_dir}/job_{job_id}.yaml'
    upload = remote.upload_file(hostname, config_path, remote_config)
    if not upload['ok']:
        return jsonify({'error': f'Upload failed: {upload["stderr"]}'}), 500

    # Start training via screen
    session_name = f'orracle_{job_id}'
    result = remote.start_remote_training(
        hostname, niftytune_dir, remote_config,
        venv=data.get('venv', 'venv_mlx'),
        session_name=session_name,
    )
    if not result.get('ok'):
        return jsonify({'error': result.get('error', 'Failed to start remote training')}), 500

    log_path = result.get('log_path', f'{niftytune_dir}/training.log')

    # Start a remote monitor for live streaming
    total_iters = int(data.get('iters', 25000))
    monitor_id = remote.start_remote_monitor(
        hostname, log_path, total_iters, job_id=job_id)

    # Register the job in jobs.yaml
    job_meta = {
        'id': job_id,
        'status': 'running',
        'framework': data.get('framework', 'mlx_lora'),
        'machine': machine_name,
        'model': data.get('model', ''),
        'output_name': data.get('output_name', f'model_{job_id}'),
        'total_iters': total_iters,
        'config': {},
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': None,
        'final_loss': None,
        'best_val_loss': None,
        'monitor_id': monitor_id,
        'remote_log': log_path,
        'session_name': session_name,
    }
    with jobs._jobs_lock:
        all_jobs = jobs._load_jobs()
        all_jobs[job_id] = job_meta
        jobs._save_jobs(all_jobs)

    return jsonify({
        'job_id': job_id,
        'monitor_id': monitor_id,
        'status': 'started',
        'log_path': log_path,
    })


@training_bp.route('/api/stop/<job_id>', methods=['POST'])
def api_stop(job_id):
    """Stop a training job."""
    force = request.json.get('force', False) if request.json else False

    # Check if remote job
    job = jobs.get_job(job_id)
    if job and job.get('machine', 'local') != 'local':
        machines = _load_machines()
        m = machines.get(job['machine'], {})
        hostname = m.get('hostname')
        if hostname:
            session = job.get('session_name', f'orracle_{job_id}')
            remote.stop_remote_training(hostname, session_name=session)
            if job.get('monitor_id'):
                remote.stop_remote_monitor(job['monitor_id'])
            jobs.update_job(job_id, {
                'status': 'cancelled',
                'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
            })
            return jsonify({'ok': True})

    ok = jobs.stop_job(job_id, force=force)
    return jsonify({'ok': ok})


@training_bp.route('/api/jobs')
def api_jobs():
    """List all training jobs."""
    return jsonify(jobs.list_jobs())


@training_bp.route('/api/job/<job_id>')
def api_job(job_id):
    """Get details for a specific job."""
    job = jobs.get_job(job_id)
    if not job:
        return jsonify({'error': 'not found'}), 404
    active = jobs.get_active_job(job_id)
    if active:
        job['_active'] = {
            'lines': len(active['output_lines']),
            'metrics': len(active['metrics']),
            'done': active['done'],
        }
    return jsonify(job)


@training_bp.route('/api/stream/<job_id>')
def api_stream(job_id):
    """SSE endpoint for live training output and metrics."""
    def generate():
        sent_lines = 0
        sent_metrics = 0

        # Wait for job to become active (up to 30s)
        waited = 0
        active = None
        while waited < 30:
            active = jobs.get_active_job(job_id)
            if active:
                break
            time.sleep(1)
            waited += 1

        if not active:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found or not active'})}\n\n"
            return

        while True:
            with jobs._active_lock:
                lines = active['output_lines']
                metrics = active['metrics']
                done = active['done']
                exit_code = active['exit_code']
                new_lines = lines[sent_lines:]
                new_metrics = metrics[sent_metrics:]
                sent_lines = len(lines)
                sent_metrics = len(metrics)

            for line in new_lines:
                yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"

            for m in new_metrics:
                yield f"data: {json.dumps({'type': 'metric', **m})}\n\n"

            if done:
                yield f"data: {json.dumps({'type': 'done', 'exit_code': exit_code})}\n\n"
                break

            # ETA calculation
            if new_metrics:
                total = active.get('total_iters', 0)
                eta = log_parser.estimate_eta(metrics, total)
                if eta is not None:
                    yield f"data: {json.dumps({'type': 'eta', 'seconds': eta})}\n\n"

            time.sleep(0.5)

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@training_bp.route('/api/metrics/<job_id>')
def api_metrics(job_id):
    """Get all metrics for a job (for chart initialization)."""
    # Try local active job
    active = jobs.get_active_job(job_id)
    if active:
        with jobs._active_lock:
            metrics = list(active['metrics'])
            done = active['done']
        chart_metrics = log_parser.downsample_metrics(metrics)
        return jsonify({
            'metrics': chart_metrics,
            'done': done,
            'total_iters': active.get('total_iters', 0),
            'best_val_loss': active.get('best_val_loss'),
        })

    # Try remote monitor
    job = jobs.get_job(job_id)
    if job and job.get('monitor_id'):
        monitor = remote.get_remote_monitor(job['monitor_id'])
        if monitor:
            with remote._remote_lock:
                metrics = list(monitor['metrics'])
                done = monitor['done']
            chart_metrics = log_parser.downsample_metrics(metrics)
            return jsonify({
                'metrics': chart_metrics,
                'done': done,
                'total_iters': monitor.get('total_iters', 0),
                'best_val_loss': monitor.get('best_val_loss'),
            })

    return jsonify({'metrics': [], 'done': True})


@training_bp.route('/api/reattach/<job_id>', methods=['POST'])
def api_reattach(job_id):
    """Re-attach a remote monitor to a running job (e.g. after server restart)."""
    job = jobs.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.get('machine', 'local') == 'local':
        return jsonify({'error': 'Only remote jobs can be re-attached'}), 400

    machines = _load_machines()
    m = machines.get(job['machine'], {})
    hostname = m.get('hostname')
    if not hostname:
        return jsonify({'error': f'Machine {job["machine"]} not found'}), 404

    log_path = job.get('remote_log')
    if not log_path:
        return jsonify({'error': 'No remote log path for this job'}), 400

    # Check if training is still running on remote
    running = remote.detect_remote_training(hostname)
    if not running:
        jobs.update_job(job_id, {
            'status': 'completed',
            'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        return jsonify({'ok': True, 'status': 'completed',
                        'message': 'Training already finished'})

    # Start a new monitor
    monitor_id = remote.start_remote_monitor(
        hostname, log_path, job.get('total_iters', 0), job_id=job_id)
    if not monitor_id:
        return jsonify({'error': 'Failed to start monitor'}), 500

    jobs.update_job(job_id, {'monitor_id': monitor_id})
    return jsonify({'ok': True, 'monitor_id': monitor_id})
