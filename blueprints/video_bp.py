"""Video training blueprint — key-moment detector UI and job management."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time

from flask import (Blueprint, Response, jsonify, render_template,
                   request, stream_with_context)

from training import jobs as _jobs

video_bp = Blueprint('video', __name__, url_prefix='/workshop/video')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VIDEO_TRAIN_SCRIPT = os.path.join(_PROJECT_ROOT, 'training', 'video', 'train.py')
_VIDEO_CONFIG = os.path.join(_PROJECT_ROOT, 'training', 'video', 'config.yaml')
_CKPT_DIR = os.path.join(_PROJECT_ROOT, 'output', 'video_training', 'ckpt')

# ---------------------------------------------------------------------------
# Active video job tracking (separate from text training jobs)
# ---------------------------------------------------------------------------

_video_jobs: dict = {}   # job_id -> {process, output_lines, metrics, done, exit_code}
_video_lock = threading.Lock()


def _get_checkpoint_info() -> dict:
    """Return info about the latest checkpoint, or empty dict if none."""
    if not os.path.isdir(_CKPT_DIR):
        return {}
    last = os.path.join(_CKPT_DIR, 'last.pt')
    if not os.path.isfile(last):
        # Look for numbered checkpoints
        ckpts = sorted(f for f in os.listdir(_CKPT_DIR)
                       if f.startswith('ckpt_') and f.endswith('.pt'))
        if not ckpts:
            return {}
        last = os.path.join(_CKPT_DIR, ckpts[-1])

    stat = os.stat(last)
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
    size_mb = round(stat.st_size / 1024 / 1024, 1)

    # Try to read step from checkpoint metadata (non-blocking)
    step = None
    try:
        import torch
        ck = torch.load(last, map_location='cpu', weights_only=True)
        step = int(ck.get('step', 0))
    except Exception:
        pass

    return {
        'path': last,
        'mtime': mtime,
        'size_mb': size_mb,
        'step': step,
    }


def _parse_video_line(line: str) -> dict | None:
    """Parse a log line from train.py and return a metric dict or None."""
    # Format: "step 500/200000 | loss 0.2345 | acc 0.8900 | lr 9.99e-05 | 12.34 it/s"
    if line.startswith('step ') and '|' in line:
        try:
            parts = {p.strip().split()[0]: p.strip().split()[1]
                     for p in line.split('|')}
            step_part = line.split('|')[0].strip()  # "step 500/200000"
            step_tok = step_part.split()
            step = int(step_tok[1].split('/')[0])
            total = int(step_tok[1].split('/')[1])
            return {
                'step': step,
                'total': total,
                'loss': float(parts.get('loss', 0)),
                'acc': float(parts.get('acc', 0)),
                'lr': float(parts.get('lr', 0)),
            }
        except Exception:
            return None
    # Format: "[val step 500] loss=0.1234 acc=0.9100 ..."
    if line.startswith('[val step '):
        try:
            step = int(line.split('[val step ')[1].split(']')[0])
            kv_str = line.split(']', 1)[1].strip()
            kvs = {}
            for tok in kv_str.split():
                k, _, v = tok.partition('=')
                kvs[k] = float(v)
            return {'type': 'val', 'step': step, **kvs}
        except Exception:
            return None
    return None


def _read_video_output(job_id: str) -> None:
    """Background thread: read video trainer stdout and parse metrics."""
    with _video_lock:
        active = _video_jobs.get(job_id)
    if not active:
        return

    proc = active['process']
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip('\n')
            if not line.strip():
                continue
            with _video_lock:
                active['output_lines'].append(line)
                m = _parse_video_line(line)
                if m:
                    active['metrics'].append(m)
                    if m.get('type') == 'val' and m.get('loss'):
                        bvl = active.get('best_val_loss')
                        if bvl is None or m['loss'] < bvl:
                            active['best_val_loss'] = m['loss']
    except Exception:
        pass

    ret = proc.wait()
    with _video_lock:
        active['done'] = True
        active['exit_code'] = ret

    status = 'completed' if ret == 0 else 'failed'
    _jobs.update_job(job_id, {
        'status': status,
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
    })


def _start_video_process(job_id: str, resume: bool = False) -> str:
    """Launch train.py subprocess. Returns job_id."""
    cmd = [sys.executable, '-u', _VIDEO_TRAIN_SCRIPT,
           '--config', _VIDEO_CONFIG]
    if resume:
        cmd.append('--resume')

    job_meta = {
        'id': job_id,
        'status': 'starting',
        'framework': 'video_detector',
        'machine': 'local',
        'model': 'KeyMomentDetector',
        'output_name': 'video_detector',
        'total_iters': 200_000,
        'config': {'resume': resume},
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': None,
        'final_loss': None,
        'best_val_loss': None,
    }
    with _jobs._jobs_lock:
        all_jobs = _jobs._load_jobs()
        all_jobs[job_id] = job_meta
        _jobs._save_jobs(all_jobs)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=_PROJECT_ROOT,
        text=True,
        bufsize=1,
    )

    active = {
        'process': proc,
        'output_lines': [],
        'metrics': [],
        'done': False,
        'exit_code': 0,
        'total_iters': 200_000,
        'best_val_loss': None,
    }
    with _video_lock:
        _video_jobs[job_id] = active

    _jobs.update_job(job_id, {'status': 'running', 'pid': proc.pid})

    t = threading.Thread(target=_read_video_output, args=(job_id,), daemon=True)
    t.start()

    return job_id


def _get_running_video_job() -> dict | None:
    """Return the active video job record if one is running, else None."""
    with _video_lock:
        for jid, active in _video_jobs.items():
            if not active['done']:
                job = _jobs.get_job(jid)
                if job:
                    return job
    # Also check persisted jobs for any 'running' video job
    for job in _jobs.list_jobs():
        if job.get('framework') == 'video_detector' and job.get('status') == 'running':
            return job
    return None


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@video_bp.route('/')
@video_bp.route('')
def index():
    """Video training main page."""
    ckpt = _get_checkpoint_info()
    running = _get_running_video_job()
    recent_jobs = [j for j in _jobs.list_jobs()
                   if j.get('framework') == 'video_detector'][:10]
    return render_template(
        'workshop/video.html',
        ckpt=ckpt,
        running_job=running,
        recent_jobs=recent_jobs,
    )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@video_bp.route('/api/start', methods=['POST'])
def api_start():
    """Start or resume a video training job."""
    data = request.json or {}
    resume = bool(data.get('resume', False))

    running = _get_running_video_job()
    if running:
        return jsonify({'error': 'A video training job is already running',
                        'job_id': running['id']}), 409

    job_id = str(int(time.time()))
    try:
        _start_video_process(job_id, resume=resume)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'job_id': job_id, 'status': 'started', 'resume': resume})


@video_bp.route('/api/stop/<job_id>', methods=['POST'])
def api_stop(job_id):
    """Stop a running video training job (sends SIGTERM for clean checkpoint flush)."""
    with _video_lock:
        active = _video_jobs.get(job_id)

    if active and not active['done']:
        proc = active['process']
        proc.terminate()   # SIGTERM — train.py catches this and flushes last.pt
        # Give it 5 s to flush, then force-kill
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        _jobs.update_job(job_id, {
            'status': 'cancelled',
            'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        return jsonify({'ok': True})

    return jsonify({'ok': False, 'error': 'Job not found or already done'}), 404


@video_bp.route('/api/status')
def api_status():
    """Current checkpoint info + running job summary."""
    ckpt = _get_checkpoint_info()
    running = _get_running_video_job()
    return jsonify({'ckpt': ckpt, 'running': running})


@video_bp.route('/api/stream/<job_id>')
def api_stream(job_id):
    """SSE endpoint for live video training output."""
    def generate():
        sent_lines = 0
        sent_metrics = 0

        # Wait up to 15s for the job to appear
        waited = 0
        while waited < 15:
            with _video_lock:
                active = _video_jobs.get(job_id)
            if active:
                break
            time.sleep(1)
            waited += 1

        if not active:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
            return

        while True:
            with _video_lock:
                lines = list(active['output_lines'])
                metrics = list(active['metrics'])
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

            time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@video_bp.route('/api/jobs')
def api_jobs():
    """List past video training jobs."""
    jobs = [j for j in _jobs.list_jobs() if j.get('framework') == 'video_detector']
    return jsonify(jobs)
