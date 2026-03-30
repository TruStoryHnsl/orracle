"""Text generation blueprint — clean chat interface for Ollama models.

Supports single-panel chat with any network Ollama instance.
Streaming via SSE, model profiles from model_registry.yaml.
"""

from __future__ import annotations

import json

from flask import (Blueprint, Response, current_app, jsonify,
                   render_template, request, stream_with_context)

from training import generate

text_gen_bp = Blueprint('text_gen', __name__, url_prefix='/studio/text')


@text_gen_bp.route('/')
def index():
    """Text generation chat interface."""
    models = generate.list_all_network_models()
    registry = generate.load_model_registry()
    profiles = registry.get('profiles', {})
    return render_template('studio/text.html',
                           models=models, profiles=profiles)


@text_gen_bp.route('/api/chat', methods=['POST'])
def api_chat():
    """SSE streaming chat endpoint — proxies to Ollama."""
    data = request.json or {}
    model = data.get('model', '').strip()
    messages = data.get('messages', [])
    options = data.get('options', {})
    host = data.get('host')

    if not model or not messages:
        return jsonify({'error': 'model and messages required'}), 400

    def stream():
        try:
            for chunk in generate.stream_chat(model, messages, options, host=host):
                if isinstance(chunk, dict) and '_stats' in chunk:
                    yield f"data: {json.dumps({'type': 'stats', **chunk['_stats']})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(stream()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@text_gen_bp.route('/api/models')
def api_models():
    """List all network Ollama models."""
    return jsonify(generate.list_all_network_models())
