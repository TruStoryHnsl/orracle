"""Image generation blueprint — ComfyUI frontend.

Connects to any online ComfyUI instance via the service manager.
Provides prompt interface, parameter controls, and output gallery.
Server builds workflows from parameters — clients never send raw ComfyUI JSON.
"""

from __future__ import annotations

from flask import (Blueprint, Response, current_app, jsonify,
                   render_template, request)

from shared import load_yaml, save_yaml
from training import comfyui

image_gen_bp = Blueprint('image_gen', __name__, url_prefix='/studio/image')


def _get_comfyui_url():
    """Get URL of an online ComfyUI instance from service manager."""
    svc_mgr = current_app.config.get('service_manager')
    if svc_mgr:
        online = svc_mgr.find_online('comfyui')
        if online:
            return online[0]['url']
    return comfyui.COMFYUI_DEFAULT


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@image_gen_bp.route('/')
def index():
    """Image generation interface."""
    url = _get_comfyui_url()
    available = comfyui.comfyui_available(url)
    checkpoints = comfyui.list_checkpoints(url) if available else []
    loras = comfyui.list_loras(url) if available else []
    samplers = comfyui.list_samplers(url) if available else []
    schedulers = comfyui.list_schedulers(url) if available else []

    svc_mgr = current_app.config.get('service_manager')
    instances = svc_mgr.get_by_type('comfyui') if svc_mgr else []

    profiles = load_yaml('image_profiles.yaml').get('profiles', {})

    return render_template('studio/image.html',
                           comfyui_url=url,
                           comfyui_available=available,
                           checkpoints=checkpoints,
                           loras=loras,
                           samplers=samplers,
                           schedulers=schedulers,
                           instances=instances,
                           profiles=profiles)


# ---------------------------------------------------------------------------
# API — discovery
# ---------------------------------------------------------------------------

@image_gen_bp.route('/api/status')
def api_status():
    url = request.args.get('url') or _get_comfyui_url()
    available = comfyui.comfyui_available(url)
    stats = comfyui.get_system_stats(url) if available else None
    return jsonify({'available': available, 'url': url, 'stats': stats})


@image_gen_bp.route('/api/checkpoints')
def api_checkpoints():
    url = request.args.get('url') or _get_comfyui_url()
    return jsonify(comfyui.list_checkpoints(url))


@image_gen_bp.route('/api/loras')
def api_loras():
    url = request.args.get('url') or _get_comfyui_url()
    return jsonify(comfyui.list_loras(url))


@image_gen_bp.route('/api/samplers')
def api_samplers():
    url = request.args.get('url') or _get_comfyui_url()
    return jsonify(comfyui.list_samplers(url))


@image_gen_bp.route('/api/schedulers')
def api_schedulers():
    url = request.args.get('url') or _get_comfyui_url()
    return jsonify(comfyui.list_schedulers(url))


# ---------------------------------------------------------------------------
# API — generation (server-side workflow building)
# ---------------------------------------------------------------------------

@image_gen_bp.route('/api/generate', methods=['POST'])
def api_generate():
    """Build workflow from parameters and submit to ComfyUI.

    Accepts generation parameters (not raw workflow).
    Returns prompt_id for polling.
    """
    data = request.json or {}
    url = data.get('url') or _get_comfyui_url()

    checkpoint = data.get('checkpoint')
    if not checkpoint:
        return jsonify({'error': 'checkpoint required'}), 400

    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400

    seed = data.get('seed')
    if seed is not None and seed < 0:
        seed = None  # negative = random

    workflow = comfyui.build_workflow(
        checkpoint=checkpoint,
        loras=data.get('loras', []),
        prompt=prompt,
        negative_prompt=data.get('negative_prompt', ''),
        seed=seed,
        steps=data.get('steps', 20),
        cfg=data.get('cfg', 7.0),
        width=data.get('width', 1024),
        height=data.get('height', 1024),
        sampler=data.get('sampler', 'euler'),
        scheduler=data.get('scheduler', 'normal'),
        batch_size=data.get('batch_size', 1),
        clip_skip=data.get('clip_skip', 1),
        filename_prefix='orracle/' + checkpoint.split('.')[0][:20],
    )

    prompt_id = comfyui.queue_prompt(url, workflow)
    if not prompt_id:
        return jsonify({'error': 'Failed to queue prompt — is ComfyUI running?'}), 502

    return jsonify({'prompt_id': prompt_id})


@image_gen_bp.route('/api/poll/<prompt_id>')
def api_poll(prompt_id):
    """Poll for generation completion. Returns image info when done."""
    url = request.args.get('url') or _get_comfyui_url()
    history = comfyui.get_history(url, prompt_id)

    if not history or prompt_id not in history:
        return jsonify({'status': 'pending'})

    entry = history[prompt_id]
    images = comfyui.extract_images(entry)
    return jsonify({
        'status': 'done',
        'images': images,
    })


# ---------------------------------------------------------------------------
# API — image proxy (serves ComfyUI images, avoids CORS)
# ---------------------------------------------------------------------------

@image_gen_bp.route('/api/proxy')
def api_proxy():
    """Proxy an image from ComfyUI. Avoids CORS and keeps ComfyUI internal."""
    url = request.args.get('url') or _get_comfyui_url()
    filename = request.args.get('filename', '')
    subfolder = request.args.get('subfolder', '')
    img_type = request.args.get('type', 'output')

    if not filename:
        return '', 400

    img_bytes = comfyui.fetch_image(url, filename, subfolder, img_type)
    if not img_bytes:
        return '', 404

    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'png'
    mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'webp': 'image/webp'}.get(ext, 'image/png')

    return Response(img_bytes, mimetype=mime,
                    headers={'Cache-Control': 'public, max-age=3600'})


# ---------------------------------------------------------------------------
# API — ComfyUI queue report (orracle jobs only)
# ---------------------------------------------------------------------------

@image_gen_bp.route('/api/queue')
def api_queue():
    """Get ComfyUI queue status filtered to orracle-originated jobs."""
    url = request.args.get('url') or _get_comfyui_url()
    return jsonify(comfyui.get_orracle_queue(url))


# ---------------------------------------------------------------------------
# API — generation profiles (save/load/delete)
# ---------------------------------------------------------------------------

PROFILE_FIELDS = (
    'checkpoint', 'sampler', 'scheduler', 'steps', 'cfg',
    'width', 'height', 'batch_size', 'clip_skip',
    'negative_prompt', 'loras',
)


@image_gen_bp.route('/api/profiles')
def api_profiles_list():
    """List all saved profiles."""
    profiles = load_yaml('image_profiles.yaml').get('profiles', {})
    return jsonify([{'name': name, **cfg} for name, cfg in profiles.items()])


@image_gen_bp.route('/api/profiles/<name>')
def api_profile_get(name):
    """Load a single profile by name."""
    profiles = load_yaml('image_profiles.yaml').get('profiles', {})
    profile = profiles.get(name)
    if not profile:
        return jsonify({'error': 'Profile not found'}), 404
    return jsonify({'name': name, **profile})


@image_gen_bp.route('/api/profiles', methods=['POST'])
def api_profile_save():
    """Save or overwrite a named profile."""
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'name required'}), 400

    profile = {}
    for field in PROFILE_FIELDS:
        if field in data:
            profile[field] = data[field]

    all_data = load_yaml('image_profiles.yaml')
    if 'profiles' not in all_data:
        all_data['profiles'] = {}
    all_data['profiles'][name] = profile
    save_yaml('image_profiles.yaml', all_data)

    return jsonify({'ok': True, 'name': name})


@image_gen_bp.route('/api/profiles/<name>', methods=['DELETE'])
def api_profile_delete(name):
    """Delete a profile."""
    all_data = load_yaml('image_profiles.yaml')
    profiles = all_data.get('profiles', {})
    if name not in profiles:
        return jsonify({'error': 'Profile not found'}), 404
    del profiles[name]
    save_yaml('image_profiles.yaml', all_data)
    return jsonify({'ok': True})
