"""Forge blueprint — LoRA weight refinement, evolutionary passes, tournaments."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, render_template, request

from training import comfyui, forge

forge_bp = Blueprint('forge', __name__, url_prefix='/forge')


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@forge_bp.route('/')
def index():
    """Forge workspace — evolutionary LoRA weight refinement."""
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    available = comfyui.comfyui_available(url)
    checkpoints = comfyui.list_checkpoints(url) if available else []
    loras = comfyui.list_loras(url) if available else []
    samplers = comfyui.list_samplers(url) if available else []
    schedulers = comfyui.list_schedulers(url) if available else []
    projects = forge.list_projects()
    return render_template('forge/forge.html',
                           comfyui_url=url,
                           comfyui_available=available,
                           checkpoints=checkpoints,
                           loras=loras,
                           samplers=samplers,
                           schedulers=schedulers,
                           projects=projects)


# ---------------------------------------------------------------------------
# ComfyUI API routes
# ---------------------------------------------------------------------------

@forge_bp.route('/api/comfyui/status')
def api_comfyui_status():
    """Check ComfyUI availability and system stats."""
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    available = comfyui.comfyui_available(url)
    stats = comfyui.get_system_stats(url) if available else None
    return jsonify({'available': available, 'stats': stats})


@forge_bp.route('/api/comfyui/checkpoints')
def api_comfyui_checkpoints():
    """List available ComfyUI checkpoints."""
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    return jsonify(comfyui.list_checkpoints(url))


@forge_bp.route('/api/comfyui/loras')
def api_comfyui_loras():
    """List available ComfyUI LoRAs."""
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    return jsonify(comfyui.list_loras(url))


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

@forge_bp.route('/api/projects')
def api_projects():
    """List all forge projects."""
    return jsonify(forge.list_projects())


@forge_bp.route('/api/projects', methods=['POST'])
def api_create_project():
    """Create a new forge project."""
    data = request.json or {}
    name = data.get('name', '').strip()
    checkpoint = data.get('checkpoint', '').strip()
    loras = data.get('loras', [])
    if not name or not checkpoint or not loras:
        return jsonify({'error': 'name, checkpoint, and loras required'}), 400
    defaults = data.get('defaults', {})
    project = forge.create_project(name, checkpoint, loras, defaults)
    return jsonify({'id': project['id'], 'name': project['name']})


@forge_bp.route('/api/project/<project_id>')
def api_project(project_id):
    """Get a forge project with tree summary."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    project['tree_summary'] = forge.get_tree_summary(project)
    return jsonify(project)


@forge_bp.route('/api/project/<project_id>', methods=['DELETE'])
def api_delete_project(project_id):
    """Delete a forge project."""
    ok = forge.delete_project(project_id)
    return jsonify({'ok': ok})


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

@forge_bp.route('/api/project/<project_id>/navigate', methods=['POST'])
def api_navigate(project_id):
    """Navigate to a node in the project tree."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    data = request.json or {}
    path = data.get('path', '')
    ok = forge.navigate(project, path)
    if not ok:
        return jsonify({'error': 'invalid path'}), 400
    forge.save_project(project)
    return jsonify({'ok': True, 'current_node': project['current_node']})


# ---------------------------------------------------------------------------
# Evolutionary passes
# ---------------------------------------------------------------------------

@forge_bp.route('/api/project/<project_id>/pass', methods=['POST'])
def api_start_pass(project_id):
    """Start an evolutionary pass (sweep/refine/polish)."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    data = request.json or {}
    tier = data.get('tier', 'sweep')
    lora_focus = data.get('lora_focus', '')
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    if not lora_focus:
        return jsonify({'error': 'lora_focus required'}), 400
    result = forge.start_pass(project, tier, lora_focus, comfyui_url=url)
    if not result:
        return jsonify({'error': 'failed to start pass'}), 500
    return jsonify(result)


@forge_bp.route('/api/project/<project_id>/pass/<int:pass_num>')
def api_pass_status(project_id, pass_num):
    """Get status of an evolutionary pass."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    if pass_num < 1 or pass_num > len(project.get('passes', [])):
        return jsonify({'error': 'invalid pass number'}), 404
    return jsonify(project['passes'][pass_num - 1])


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@forge_bp.route('/api/project/<project_id>/pass/<int:pass_num>/feedback', methods=['POST'])
def api_feedback(project_id, pass_num):
    """Submit feedback (keep/trash/neutral) for a pass's candidates."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    data = request.json or {}
    labels = data.get('labels', {})
    ok = forge.submit_feedback(project, pass_num, labels)
    if not ok:
        return jsonify({'error': 'feedback submission failed'}), 400
    return jsonify({'ok': True})


# ---------------------------------------------------------------------------
# Promote & Branch
# ---------------------------------------------------------------------------

@forge_bp.route('/api/project/<project_id>/promote', methods=['POST'])
def api_promote(project_id):
    """Promote a LoRA by baking it into a new checkpoint."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    data = request.json or {}
    lora_name = data.get('lora_name', '')
    model_weight = float(data.get('model_weight', 0.5))
    clip_weight = float(data.get('clip_weight', 0.5))
    branch_name = data.get('branch_name', '')
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    result = forge.promote_lora(project, lora_name, model_weight, clip_weight,
                                branch_name, comfyui_url=url)
    if result.get('error'):
        return jsonify(result), 400
    return jsonify(result)


@forge_bp.route('/api/project/<project_id>/branch', methods=['POST'])
def api_branch(project_id):
    """Create a development branch in the project tree."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    data = request.json or {}
    name = data.get('name', '')
    from_path = data.get('from_path')
    result = forge.create_branch(project, name, from_path)
    if result.get('error'):
        return jsonify(result), 400
    return jsonify(result)


# ---------------------------------------------------------------------------
# Tournament mode
# ---------------------------------------------------------------------------

@forge_bp.route('/api/project/<project_id>/tournament', methods=['POST'])
def api_tournament(project_id):
    """Start a tournament between two branches."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    data = request.json or {}
    branch_a = data.get('branch_a', '')
    branch_b = data.get('branch_b', '')
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    if not branch_a or not branch_b:
        return jsonify({'error': 'branch_a and branch_b required'}), 400
    result = forge.start_tournament(project, branch_a, branch_b, comfyui_url=url)
    if result.get('error'):
        return jsonify(result), 400
    return jsonify(result)


@forge_bp.route('/api/project/<project_id>/tournament/<int:tournament_id>/vote', methods=['POST'])
def api_tournament_vote(project_id, tournament_id):
    """Submit votes for a tournament round."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    data = request.json or {}
    votes = data.get('votes', {})
    ok = forge.submit_tournament_vote(project, tournament_id, votes)
    if not ok:
        return jsonify({'error': 'vote submission failed'}), 400
    return jsonify({'ok': True})


# ---------------------------------------------------------------------------
# Image proxying
# ---------------------------------------------------------------------------

@forge_bp.route('/api/image/<project_id>/<int:pass_num>/<candidate_id>')
def api_image(project_id, pass_num, candidate_id):
    """Proxy an image from ComfyUI for a forge candidate."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    result = forge.get_candidate_image(project, pass_num, candidate_id)
    if not result:
        return jsonify({'error': 'image not found'}), 404
    filename, subfolder = result
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    img_bytes = comfyui.fetch_image(url, filename, subfolder)
    if img_bytes:
        return Response(img_bytes, mimetype='image/png')
    return jsonify({'error': 'image fetch failed'}), 500


@forge_bp.route('/api/tournament-image/<project_id>/<int:tournament_id>/<int:pair_idx>/<side>')
def api_tournament_image(project_id, tournament_id, pair_idx, side):
    """Proxy a tournament image from ComfyUI."""
    project = forge.load_project(project_id)
    if not project:
        return jsonify({'error': 'not found'}), 404
    result = forge.get_tournament_image(project, tournament_id, pair_idx, side)
    if not result:
        return jsonify({'error': 'image not found'}), 404
    filename, subfolder = result
    url = request.args.get('url', comfyui.COMFYUI_DEFAULT)
    img_bytes = comfyui.fetch_image(url, filename, subfolder)
    if img_bytes:
        return Response(img_bytes, mimetype='image/png')
    return jsonify({'error': 'image fetch failed'}), 500


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@forge_bp.route('/api/project/<project_id>/history')
def api_history(project_id):
    """Download the generation history CSV for a project."""
    csv_path = forge.get_history_csv_path(project_id)
    if not csv_path:
        return jsonify({'error': 'no history yet'}), 404
    with open(csv_path) as f:
        content = f.read()
    return Response(content, mimetype='text/csv',
                    headers={'Content-Disposition':
                             f'attachment; filename={project_id}_history.csv'})


@forge_bp.route('/api/project/<project_id>/history.json')
def api_history_json(project_id):
    """Get generation history as JSON."""
    rows = forge.load_history(project_id)
    return jsonify(rows)
