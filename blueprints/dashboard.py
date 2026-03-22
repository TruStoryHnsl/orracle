"""Dashboard blueprint — unified home page showing all active work."""

from __future__ import annotations

import os

import yaml
from flask import Blueprint, jsonify, render_template

from training import jobs, hardware, remote
from training import export_mgr as export

dashboard_bp = Blueprint('dashboard', __name__)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_machines() -> dict:
    try:
        with open(os.path.join(CONFIG_DIR, 'machines.yaml')) as f:
            data = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        data = {}
    return data.get('machines', {})


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@dashboard_bp.route('/')
def index():
    """Unified dashboard — active jobs, machine status, quick links."""
    hw = hardware.detect_hardware()
    machines = _load_machines()
    all_jobs = jobs.list_jobs()

    # Enrich active jobs with latest metrics
    active_jobs = []
    for job in all_jobs:
        if job['status'] == 'running':
            # Try local active job first
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
            # Try remote monitor for remote jobs
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
            active_jobs.append(job)

    recent_jobs = [j for j in all_jobs
                   if j['status'] in ('completed', 'failed', 'cancelled')][:10]

    # Ollama models for quick glance
    try:
        ollama_models = export.list_ollama_models()
    except Exception:
        ollama_models = []

    return render_template('trainer_dashboard.html',
                           hw=hw,
                           machines=machines,
                           active_jobs=active_jobs,
                           recent_jobs=recent_jobs,
                           ollama_models=ollama_models,
                           gpu_summary=hardware.format_gpu_summary(hw),
                           ram_summary=hardware.format_ram_summary(hw))


@dashboard_bp.route('/api/status')
def api_status():
    """Quick status API for dashboard widgets."""
    all_jobs = jobs.list_jobs()
    active = [j for j in all_jobs if j['status'] == 'running']
    recent = [j for j in all_jobs
              if j['status'] in ('completed', 'failed', 'cancelled')][:5]
    machines = _load_machines()
    monitors = remote.list_remote_monitors()

    return jsonify({
        'active_jobs': len(active),
        'recent_jobs': len(recent),
        'machines': len(machines),
        'active_monitors': len([m for m in monitors if not m.get('done')]),
    })
