"""Image generation blueprint — ComfyUI frontend.

Connects to any online ComfyUI instance via the service manager.
Provides prompt interface, parameter controls, and output gallery.
Server builds workflows from parameters — clients never send raw ComfyUI JSON.
"""

from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path

from flask import (Blueprint, Response, current_app, jsonify,
                   render_template, request, session)

from shared import load_yaml, save_yaml, CONFIG_DIR
from training import comfyui

image_gen_bp = Blueprint('image_gen', __name__, url_prefix='/studio/image')

# ---------------------------------------------------------------------------
# Session history — in-memory, keyed by Flask session id
# ---------------------------------------------------------------------------

_history_store: dict[str, list[dict]] = {}
_history_lock = threading.Lock()
MAX_HISTORY = 100  # per session


def _session_key() -> str:
    """Return a stable key for the current Flask session."""
    # Use the Flask session's sid if available, else create a server-side key.
    from flask import session as flask_session
    key = flask_session.get('_img_history_key')
    if not key:
        import uuid
        key = str(uuid.uuid4())
        flask_session['_img_history_key'] = key
        flask_session.permanent = True
    return key


def _add_to_history(entry: dict):
    """Append a generation result to the current session's history."""
    key = _session_key()
    with _history_lock:
        bucket = _history_store.setdefault(key, [])
        bucket.insert(0, entry)  # newest first
        if len(bucket) > MAX_HISTORY:
            del bucket[MAX_HISTORY:]


def _get_history() -> list[dict]:
    """Return current session's history (newest first)."""
    key = _session_key()
    with _history_lock:
        return list(_history_store.get(key, []))


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

    # Pre-seed history entry; images will be filled in when the client polls
    # and calls /api/history/record once generation completes.
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
# API — parameter sweep
# ---------------------------------------------------------------------------

SWEEP_PARAMS = ('seed', 'cfg', 'steps')
MAX_SWEEP_COUNT = 16


@image_gen_bp.route('/api/sweep', methods=['POST'])
def api_sweep():
    """Queue N generations varying one parameter across a list of values.

    Body:
        {same as /api/generate} plus:
        sweep_param: "seed" | "cfg" | "steps"
        sweep_values: [v1, v2, ...]   (max 16)

    Returns:
        {prompt_ids: [...]}  — one per sweep value
    """
    data = request.json or {}
    url = data.get('url') or _get_comfyui_url()

    checkpoint = data.get('checkpoint')
    if not checkpoint:
        return jsonify({'error': 'checkpoint required'}), 400
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400

    sweep_param = data.get('sweep_param', 'seed')
    if sweep_param not in SWEEP_PARAMS:
        return jsonify({'error': f'sweep_param must be one of: {SWEEP_PARAMS}'}), 400

    sweep_values = data.get('sweep_values', [])
    if not sweep_values:
        return jsonify({'error': 'sweep_values required'}), 400
    sweep_values = sweep_values[:MAX_SWEEP_COUNT]

    prompt_ids = []
    errors = []

    base_seed = data.get('seed')
    import random as _random

    for val in sweep_values:
        # Build per-sweep params
        seed = base_seed if sweep_param != 'seed' else (
            int(val) if val is not None and int(val) >= 0 else None
        )
        if seed is None and sweep_param != 'seed':
            seed = _random.randint(0, 2**32 - 1)

        cfg = float(val) if sweep_param == 'cfg' else float(data.get('cfg', 7.0))
        steps = int(val) if sweep_param == 'steps' else int(data.get('steps', 20))

        workflow = comfyui.build_workflow(
            checkpoint=checkpoint,
            loras=data.get('loras', []),
            prompt=prompt,
            negative_prompt=data.get('negative_prompt', ''),
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=data.get('width', 1024),
            height=data.get('height', 1024),
            sampler=data.get('sampler', 'euler'),
            scheduler=data.get('scheduler', 'normal'),
            batch_size=1,
            clip_skip=data.get('clip_skip', 1),
            filename_prefix='orracle/sweep_' + checkpoint.split('.')[0][:16],
        )

        pid = comfyui.queue_prompt(url, workflow)
        if pid:
            prompt_ids.append({'prompt_id': pid, 'sweep_value': val})
        else:
            errors.append(f'Failed to queue sweep value {val}')

    if not prompt_ids:
        return jsonify({'error': 'All sweep jobs failed to queue', 'errors': errors}), 502

    return jsonify({
        'prompt_ids': prompt_ids,
        'sweep_param': sweep_param,
        'count': len(prompt_ids),
        'errors': errors,
    })


# ---------------------------------------------------------------------------
# API — upscale
# ---------------------------------------------------------------------------

@image_gen_bp.route('/api/upscale', methods=['POST'])
def api_upscale():
    """Build and submit a 2× upscale workflow for an existing image.

    Body:
        filename: str      — ComfyUI output filename
        subfolder: str     — optional subfolder
        type: str          — output type (default "output")
        scale: float       — upscale factor (default 2.0)
        upscale_model: str — upscale model name (default "RealESRGAN_x4plus.pth")

    Returns:
        {prompt_id: str}
    """
    data = request.json or {}
    url = data.get('url') or _get_comfyui_url()

    filename = data.get('filename', '').strip()
    if not filename:
        return jsonify({'error': 'filename required'}), 400

    subfolder = data.get('subfolder', '')
    img_type = data.get('type', 'output')
    upscale_model = data.get('upscale_model', 'RealESRGAN_x4plus.pth')

    workflow = comfyui.build_upscale_workflow(
        filename=filename,
        subfolder=subfolder,
        img_type=img_type,
        upscale_model=upscale_model,
        filename_prefix='orracle/upscale',
    )

    prompt_id = comfyui.queue_prompt(url, workflow)
    if not prompt_id:
        return jsonify({'error': 'Failed to queue upscale — is ComfyUI running?'}), 502

    return jsonify({'prompt_id': prompt_id})


# ---------------------------------------------------------------------------
# API — session history
# ---------------------------------------------------------------------------

@image_gen_bp.route('/api/history')
def api_history_list():
    """Return this session's generation history (newest first)."""
    return jsonify(_get_history())


@image_gen_bp.route('/api/history/record', methods=['POST'])
def api_history_record():
    """Record a completed generation into session history.

    Called by the client after polling confirms completion.
    """
    data = request.json or {}
    prompt_id = data.get('prompt_id', '')
    images = data.get('images', [])
    params = data.get('params', {})

    if not prompt_id:
        return jsonify({'error': 'prompt_id required'}), 400

    entry = {
        'prompt_id': prompt_id,
        'timestamp': time.time(),
        'timestamp_str': time.strftime('%Y-%m-%d %H:%M:%S'),
        'images': images,
        'params': params,
        'thumbnail': images[0] if images else None,
    }
    _add_to_history(entry)
    return jsonify({'ok': True, 'history_size': len(_get_history())})


@image_gen_bp.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    """Clear this session's generation history."""
    key = _session_key()
    with _history_lock:
        _history_store.pop(key, None)
    return jsonify({'ok': True})


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


# ---------------------------------------------------------------------------
# Gallery — persistent, disk-backed, orracle-namespaced
# ---------------------------------------------------------------------------

RATINGS_FILE = os.path.join(CONFIG_DIR, 'image_ratings.jsonl')
_ratings_lock = threading.Lock()

# Image extensions recognised by the gallery
_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}


def _resolve_gallery_dir() -> Path | None:
    """Return the local path to the orracle output directory, or None.

    Priority: LOCAL_OUTPUT_DIR/orracle (default ComfyUI layout on this machine).
    The orracle filename prefix puts images in output/orracle/ on each machine.
    """
    base = comfyui.LOCAL_OUTPUT_DIR
    candidate = base / 'orracle'
    if candidate.exists():
        return candidate
    # Fallback: vault symlink target / orracle
    vault_candidate = comfyui.VAULT_OUTPUT_DIR / 'orracle'
    if vault_candidate.exists():
        return vault_candidate
    return None


def _load_ratings() -> dict:
    """Load all ratings from JSONL file. Returns {filename: rating} dict."""
    ratings = {}
    try:
        with _ratings_lock:
            with open(RATINGS_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get('filename') and entry.get('rating') in ('up', 'down'):
                            ratings[entry['filename']] = entry['rating']
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return ratings


def _save_rating(filename: str, rating: str | None):
    """Append a rating entry to the JSONL file.

    A None rating clears (tombstone approach — last write wins on load).
    """
    entry = {
        'filename': filename,
        'rating': rating,
        'ts': time.time(),
    }
    with _ratings_lock:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(RATINGS_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')


def _scan_gallery_dir(gallery_dir: Path, offset: int = 0, limit: int = 50,
                      date_from: str = '', date_to: str = '',
                      prefix: str = '', rating_filter: str = '',
                      ratings: dict = None) -> tuple[list, int]:
    """Scan directory for images. Returns (items, total_count).

    items are sorted newest-first.  Each item:
        {filename, subfolder, type, ts, ts_str, size, rating}
    """
    if ratings is None:
        ratings = {}

    try:
        entries = [
            e for e in gallery_dir.iterdir()
            if e.is_file() and e.suffix.lower() in _IMG_EXTS
        ]
    except OSError:
        return [], 0

    # Apply filename prefix filter
    if prefix:
        entries = [e for e in entries if e.name.startswith(prefix)]

    # Build item dicts with stat info
    items = []
    for e in entries:
        try:
            st = e.stat()
            ts = st.st_mtime
        except OSError:
            ts = 0
        items.append({
            'filename': e.name,
            'subfolder': 'orracle',
            'type': 'output',
            'ts': ts,
            'ts_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)),
            'date': time.strftime('%Y-%m-%d', time.localtime(ts)),
            'size': st.st_size if ts else 0,
            'rating': ratings.get(e.name),
        })

    # Date range filter
    if date_from:
        items = [i for i in items if i['date'] >= date_from]
    if date_to:
        items = [i for i in items if i['date'] <= date_to]

    # Rating filter: 'rated' = has any rating, 'unrated' = no rating, 'up'/'down' = specific
    if rating_filter == 'rated':
        items = [i for i in items if i['rating']]
    elif rating_filter == 'unrated':
        items = [i for i in items if not i['rating']]
    elif rating_filter in ('up', 'down'):
        items = [i for i in items if i['rating'] == rating_filter]

    # Sort newest first
    items.sort(key=lambda x: x['ts'], reverse=True)

    total = len(items)
    page = items[offset:offset + limit]
    return page, total


@image_gen_bp.route('/gallery')
def gallery():
    """Persistent image gallery — orracle-namespaced ComfyUI output."""
    gallery_dir = _resolve_gallery_dir()
    return render_template(
        'studio/gallery.html',
        gallery_dir=str(gallery_dir) if gallery_dir else None,
        gallery_available=gallery_dir is not None,
    )


@image_gen_bp.route('/api/gallery')
def api_gallery():
    """Return paginated gallery images with rating data.

    Query params:
        offset  int     (default 0)
        limit   int     (default 50, max 200)
        from    str     date YYYY-MM-DD
        to      str     date YYYY-MM-DD
        prefix  str     filename prefix filter
        rating  str     rated | unrated | up | down
    """
    offset = max(0, int(request.args.get('offset', 0)))
    limit = min(200, max(1, int(request.args.get('limit', 50))))
    date_from = request.args.get('from', '')
    date_to = request.args.get('to', '')
    prefix = request.args.get('prefix', '')
    rating_filter = request.args.get('rating', '')

    gallery_dir = _resolve_gallery_dir()
    if not gallery_dir:
        return jsonify({'items': [], 'total': 0, 'offset': offset,
                        'limit': limit, 'available': False})

    ratings = _load_ratings()
    items, total = _scan_gallery_dir(
        gallery_dir, offset=offset, limit=limit,
        date_from=date_from, date_to=date_to,
        prefix=prefix, rating_filter=rating_filter,
        ratings=ratings,
    )

    return jsonify({
        'items': items,
        'total': total,
        'offset': offset,
        'limit': limit,
        'available': True,
        'has_more': offset + limit < total,
    })


@image_gen_bp.route('/api/gallery/delete', methods=['POST'])
def api_gallery_delete():
    """Delete images by filename list.

    Body: {filenames: ["a.png", "b.png", ...]}
    Deletes from local gallery dir only (remote SSH not implemented here;
    orracle runs on orrgate which has the NAS mounted).
    """
    data = request.json or {}
    filenames = data.get('filenames', [])
    if not filenames:
        return jsonify({'error': 'filenames required'}), 400

    gallery_dir = _resolve_gallery_dir()
    if not gallery_dir:
        return jsonify({'error': 'Gallery directory not found'}), 404

    deleted = []
    errors = []
    for fname in filenames:
        # Sanitise — no path traversal
        safe = Path(fname).name
        if not safe or safe != fname:
            errors.append(f'Invalid filename: {fname}')
            continue
        fpath = gallery_dir / safe
        try:
            fpath.unlink()
            deleted.append(safe)
        except FileNotFoundError:
            errors.append(f'Not found: {safe}')
        except OSError as e:
            errors.append(f'Error deleting {safe}: {e}')

    return jsonify({'deleted': deleted, 'errors': errors, 'ok': len(deleted) > 0})


# ---------------------------------------------------------------------------
# API — per-image rating
# ---------------------------------------------------------------------------

@image_gen_bp.route('/api/image/rate', methods=['POST'])
def api_image_rate():
    """Persist a thumbs-up / thumbs-down rating for an image.

    Body:
        filename: str       — image filename (basename only)
        rating:   str       — "up", "down", or "" / null to clear
    """
    data = request.json or {}
    filename = (data.get('filename') or '').strip()
    rating = (data.get('rating') or '').strip().lower() or None

    if not filename:
        return jsonify({'error': 'filename required'}), 400
    if rating and rating not in ('up', 'down'):
        return jsonify({'error': 'rating must be "up", "down", or empty to clear'}), 400

    # Sanitise
    safe = Path(filename).name
    if safe != filename:
        return jsonify({'error': 'Invalid filename'}), 400

    _save_rating(safe, rating)
    return jsonify({'ok': True, 'filename': safe, 'rating': rating})
