"""Export blueprint — adapter management, GGUF conversion, Ollama model lifecycle."""

from __future__ import annotations

import os

import yaml
from flask import Blueprint, Response, jsonify, render_template, request

from training import jobs, remote
from training import export_mgr as export

export_bp = Blueprint('export', __name__, url_prefix='/export')

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

@export_bp.route('/')
def index():
    """Export management page — adapters, GGUF conversion, deploy pipeline."""
    adapters = export.list_adapters()
    gguf_files = export.list_gguf_files()
    templates = export.CHAT_TEMPLATES
    tasks = export.list_export_tasks()
    machines = _load_machines()

    # Build remote adapter list from recent jobs
    all_jobs = jobs._load_jobs()
    remote_adapters = []
    for jid, j in all_jobs.items():
        if j.get('status') == 'completed' and j.get('machine') != 'local':
            remote_adapters.append({
                'job_id': jid,
                'machine': j.get('machine', ''),
                'model': j.get('model', ''),
                'output_name': j.get('output_name', ''),
                'adapter_path': f"adapters_{jid}",
            })

    return render_template('export/export.html',
                           adapters=adapters,
                           gguf_files=gguf_files,
                           chat_templates=templates,
                           export_tasks=tasks,
                           machines=machines,
                           remote_adapters=remote_adapters)


@export_bp.route('/models')
def models_page():
    """Models management page — Ollama models, adapters, GGUF files."""
    ollama_models = export.list_ollama_models()
    running = export.running_ollama_models()
    running_names = {m['name'] for m in running}
    adapters = export.list_adapters()
    gguf_files = export.list_gguf_files()

    machines = _load_machines()
    remote_machines = {k: v for k, v in machines.items()
                       if v.get('hostname') and not v.get('is_local')}
    downloads = remote.list_download_tasks()

    return render_template('export/models.html',
                           ollama_models=ollama_models,
                           running_names=running_names,
                           adapters=adapters,
                           gguf_files=gguf_files,
                           remote_machines=remote_machines,
                           downloads=downloads)


# ---------------------------------------------------------------------------
# Adapter / GGUF API routes
# ---------------------------------------------------------------------------

@export_bp.route('/api/adapters')
def api_adapters():
    """List local LoRA adapters."""
    return jsonify(export.list_adapters())


@export_bp.route('/api/gguf')
def api_gguf_files():
    """List local GGUF files."""
    return jsonify(export.list_gguf_files())


@export_bp.route('/api/modelfile/preview', methods=['POST'])
def api_modelfile_preview():
    """Preview a generated Modelfile."""
    data = request.json or {}
    content = export.generate_modelfile(
        gguf_path=data.get('gguf_path', './model.gguf'),
        system_prompt=data.get('system_prompt'),
        template_key=data.get('template', 'mistral'),
        params=data.get('params'),
    )
    return jsonify({'modelfile': content})


@export_bp.route('/api/fuse', methods=['POST'])
def api_fuse():
    """Fuse a LoRA adapter into a base model."""
    data = request.json or {}
    adapter = data.get('adapter_path', '').strip()
    base_model = data.get('base_model', '').strip()
    output_dir = data.get('output_dir', '').strip()
    framework = data.get('framework', 'mlx')
    if not adapter or not base_model:
        return jsonify({'error': 'adapter_path and base_model required'}), 400
    if not output_dir:
        output_dir = str(export.NIFTYTUNE_DIR / 'models' / 'merged')
    task_id = export.start_fuse_task(adapter, base_model, output_dir, framework)
    return jsonify({'task_id': task_id, 'status': 'started'})


@export_bp.route('/api/gguf', methods=['POST'])
def api_gguf_convert():
    """Convert a model to GGUF format."""
    data = request.json or {}
    adapter = data.get('adapter_path', '').strip()
    output_name = data.get('output_name', '').strip()
    quant = data.get('quant_type', 'q4_k_m')
    framework = data.get('framework', 'unsloth')
    if not adapter:
        return jsonify({'error': 'adapter_path required'}), 400
    if not output_name:
        output_name = f'orrvert-{quant}'
    task_id = export.start_gguf_task(adapter, output_name, quant, framework)
    return jsonify({'task_id': task_id, 'status': 'started'})


@export_bp.route('/api/deploy', methods=['POST'])
def api_deploy():
    """Full pipeline: fuse adapter -> GGUF -> ollama create."""
    data = request.json or {}
    adapter = data.get('adapter_path', '').strip()
    base_model = data.get('base_model', '').strip()
    model_name = data.get('model_name', '').strip()
    if not adapter or not base_model or not model_name:
        return jsonify({'error': 'adapter_path, base_model, and model_name required'}), 400

    # Resolve remote machine info if specified
    hostname = None
    niftytune_dir = '~/niftytune'
    venv = 'venv_mlx'
    machine_name = data.get('machine', '').strip()
    if machine_name and machine_name != 'local':
        machines = _load_machines()
        m = machines.get(machine_name)
        if not m or not m.get('hostname'):
            return jsonify({'error': f'Machine "{machine_name}" not found'}), 404
        hostname = m['hostname']
        niftytune_dir = m.get('niftytune_path', '~/niftytune')
        venv = data.get('venv', 'venv_mlx')

    task_id = export.start_deploy_pipeline(
        adapter_path=adapter,
        base_model=base_model,
        model_name=model_name,
        hostname=hostname,
        niftytune_dir=niftytune_dir,
        venv=venv,
        framework=data.get('framework', 'mlx'),
        template_key=data.get('template', 'mistral'),
        system_prompt=data.get('system_prompt'),
        params=data.get('params'),
    )
    return jsonify({'task_id': task_id, 'status': 'started'})


@export_bp.route('/api/task/<task_id>')
def api_task(task_id):
    """Get status of an export task."""
    task = export.get_export_task(task_id)
    if not task:
        return jsonify({'error': 'not found'}), 404
    return jsonify(task)


@export_bp.route('/api/tasks')
def api_tasks():
    """List all export tasks."""
    return jsonify(export.list_export_tasks())


# ---------------------------------------------------------------------------
# Ollama API routes
# ---------------------------------------------------------------------------

@export_bp.route('/api/ollama/models')
def api_ollama_models():
    """List Ollama models and running instances."""
    models = export.list_ollama_models()
    running = export.running_ollama_models()
    return jsonify({'models': models, 'running': [m['name'] for m in running]})


@export_bp.route('/api/ollama/show/<path:name>')
def api_ollama_show(name):
    """Show the Modelfile for an Ollama model."""
    modelfile = export.show_ollama_modelfile(name)
    if modelfile is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'name': name, 'modelfile': modelfile})


@export_bp.route('/api/ollama/create', methods=['POST'])
def api_ollama_create():
    """Create an Ollama model from a Modelfile."""
    data = request.json or {}
    name = data.get('name', '').strip()
    modelfile = data.get('modelfile', '').strip()
    if not name or not modelfile:
        return jsonify({'error': 'name and modelfile required'}), 400
    result = export.create_ollama_model(name, modelfile)
    return jsonify(result)


@export_bp.route('/api/ollama/delete/<path:name>', methods=['POST'])
def api_ollama_delete(name):
    """Delete an Ollama model."""
    result = export.delete_ollama_model(name)
    return jsonify(result)


@export_bp.route('/api/ollama/copy', methods=['POST'])
def api_ollama_copy():
    """Copy (tag) an Ollama model."""
    data = request.json or {}
    source = data.get('source', '').strip()
    dest = data.get('dest', '').strip()
    if not source or not dest:
        return jsonify({'error': 'source and dest required'}), 400
    result = export.copy_ollama_model(source, dest)
    return jsonify(result)
