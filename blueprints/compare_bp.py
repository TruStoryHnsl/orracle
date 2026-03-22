"""Compare blueprint — side-by-side model comparison with streaming chat."""

from __future__ import annotations

import json

from flask import (Blueprint, Response, jsonify, redirect, render_template,
                   request, stream_with_context, url_for)

from training import generate

compare_bp = Blueprint('compare', __name__, url_prefix='/compare')


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@compare_bp.route('/')
def index():
    """Model comparison page."""
    models = generate.list_all_network_models()
    available = generate.ollama_available()
    return render_template('compare/compare.html', models=models,
                           ollama_available=available)


@compare_bp.route('/generate')
def generate_redirect():
    """Legacy redirect from /compare/generate to /compare."""
    return redirect(url_for('compare.index'))


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@compare_bp.route('/api/chat', methods=['POST'])
def api_chat():
    """SSE endpoint that proxies Ollama chat API for streaming responses."""
    data = request.json or {}
    model = data.get('model', '').strip()
    messages = data.get('messages', [])
    options = data.get('options', {})
    host = data.get('host')  # optional: target specific Ollama instance

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


@compare_bp.route('/api/models')
def api_models():
    """List local Ollama models."""
    return jsonify(generate.list_models())


@compare_bp.route('/api/models/network')
def api_network_models():
    """List models from all registered Ollama hosts."""
    return jsonify(generate.list_all_network_models())
