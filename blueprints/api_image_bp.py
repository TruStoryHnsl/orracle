"""Headless image generation API — authenticated endpoint for external clients.

Provides programmatic access to ComfyUI image generation for authorized
applications (initially orradash). All endpoints require X-Orracle-Key header.

Endpoints:
    POST /api/image/generate    — Submit a generation job
    GET  /api/image/status/<id> — Poll job progress
    GET  /api/image/result/<id> — Retrieve generated images
    GET  /api/image/profiles    — List available generation profiles
"""

from __future__ import annotations

import base64
import os

from flask import Blueprint, current_app, jsonify, request

from shared import load_yaml
from training import comfyui

api_image_bp = Blueprint('api_image', __name__, url_prefix='/api/image')


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    return os.environ.get('IMAGE_API_KEY', '')


@api_image_bp.before_request
def check_auth():
    """Reject requests without a valid X-Orracle-Key header."""
    key = _get_api_key()
    if not key:
        return jsonify({'error': 'API not configured — IMAGE_API_KEY not set'}), 503
    if request.headers.get('X-Orracle-Key', '') != key:
        return jsonify({'error': 'Unauthorized'}), 403


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profiles() -> dict:
    return load_yaml('image_profiles.yaml').get('profiles', {})


def _get_comfyui_url() -> str | None:
    """Find an online ComfyUI instance via the service manager."""
    svc_mgr = current_app.config.get('service_manager')
    if svc_mgr:
        online = svc_mgr.find_online('comfyui')
        if online:
            return online[0]['url']
    return None


def _get_job_queue():
    return current_app.config.get('job_queue')


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api_image_bp.route('/profiles')
def list_profiles():
    """List available generation profiles."""
    profiles = _load_profiles()
    return jsonify([{'name': name, **cfg} for name, cfg in profiles.items()])


@api_image_bp.route('/generate', methods=['POST'])
def generate():
    """Submit an image generation job.

    Request body:
        prompt (str, required): The generation prompt.
        profile (str): Profile name (default: "default").
        batch_size (int): Images per batch (overrides profile).
        batch_count (int): Number of batches to run (max 10).
        negative_prompt (str): Overrides profile negative prompt.
    """
    data = request.json or {}
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400

    # Resolve profile
    profile_name = data.get('profile', 'default')
    profiles = _load_profiles()
    profile = profiles.get(profile_name)
    if not profile:
        return jsonify({
            'error': f'Unknown profile: {profile_name}',
            'available': list(profiles.keys()),
        }), 400

    # Merge request overrides with profile defaults
    batch_size = data.get('batch_size', profile.get('batch_size', 1))
    batch_count = min(data.get('batch_count', 1), 10)
    negative_prompt = data.get('negative_prompt', profile.get('negative_prompt', ''))

    # Find ComfyUI
    comfyui_url = _get_comfyui_url()
    if not comfyui_url:
        return jsonify({'error': 'No ComfyUI instance available'}), 503

    # Build workflow from profile + prompt
    workflow = comfyui.build_workflow(
        checkpoint=profile['checkpoint'],
        loras=profile.get('loras', []),
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=profile.get('steps', 20),
        cfg=profile.get('cfg', 7.0),
        width=profile.get('width', 1024),
        height=profile.get('height', 1024),
        sampler=profile.get('sampler', 'euler'),
        scheduler=profile.get('scheduler', 'normal'),
        batch_size=batch_size,
        filename_prefix='orracle_api',
    )

    # Submit to job queue
    job_queue = _get_job_queue()
    if not job_queue:
        return jsonify({'error': 'Job queue not available'}), 503

    job_ids = []
    for _ in range(batch_count):
        job_id = job_queue.submit(
            category='gen_image',
            params={
                'workflow': workflow,
                'comfyui_url': comfyui_url,
                'profile': profile_name,
                'prompt': prompt,
                'batch_size': batch_size,
                'source': 'api',
            },
        )
        job_ids.append(job_id)

    if batch_count == 1:
        return jsonify({'job_id': job_ids[0]})
    return jsonify({'job_ids': job_ids})


@api_image_bp.route('/status/<job_id>')
def status(job_id):
    """Poll job progress.

    Returns:
        job_id, status (queued/running/complete/failed), progress (0-1), error.
    """
    job_queue = _get_job_queue()
    if not job_queue:
        return jsonify({'error': 'Queue not available'}), 503

    job = job_queue.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify({
        'job_id': job['id'],
        'status': job['status'],
        'progress': job['progress'],
        'progress_msg': job['progress_msg'],
        'error': job.get('error', ''),
        'submitted': job['submitted'],
        'started': job['started'],
        'finished': job['finished'],
    })


@api_image_bp.route('/result/<job_id>')
def result(job_id):
    """Retrieve generated images from a completed job.

    Query params:
        format: "base64" (default) or "url"
            base64 — image bytes encoded inline (portable, no ComfyUI access needed)
            url    — direct ComfyUI /view URLs (requires network access to ComfyUI)
    """
    job_queue = _get_job_queue()
    if not job_queue:
        return jsonify({'error': 'Queue not available'}), 503

    job = job_queue.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job['status'] != 'completed':
        return jsonify({
            'error': 'Job not completed',
            'status': job['status'],
        }), 409

    # Extract image info from ComfyUI history stored in job result
    result_data = job.get('result', {})
    prompt_id = result_data.get('prompt_id', '')
    history = result_data.get('output', {})
    images_info = comfyui.extract_images(history) if history else []

    # Resolve ComfyUI URL for fetching images
    comfyui_url = job['params'].get('comfyui_url') or _get_comfyui_url()
    fmt = request.args.get('format', 'base64')

    images = []
    for img in images_info:
        entry = {
            'filename': img['filename'],
            'subfolder': img.get('subfolder', ''),
        }
        if fmt == 'url' and comfyui_url:
            entry['url'] = comfyui.get_image_url(
                comfyui_url, img['filename'],
                img.get('subfolder', ''), img.get('type', 'output'))
        elif comfyui_url:
            img_bytes = comfyui.fetch_image(
                comfyui_url, img['filename'],
                img.get('subfolder', ''), img.get('type', 'output'))
            if img_bytes:
                entry['base64'] = base64.b64encode(img_bytes).decode()
                entry['mime'] = 'image/png'
        images.append(entry)

    return jsonify({
        'job_id': job_id,
        'prompt_id': prompt_id,
        'images': images,
    })
