"""Audit blueprint — jailbreak testing suite for post-training evaluation.

Tests models against known jailbreak prompts and reports pass/fail rates.
Runs via /workshop/audit.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid

from flask import (Blueprint, Response, current_app, jsonify,
                   render_template, request, stream_with_context)

from training import audit, generate

audit_bp = Blueprint('audit', __name__, url_prefix='/workshop/audit')

# Active audit runs (in-memory, keyed by audit_id)
_active_audits: dict[str, dict] = {}
_audit_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@audit_bp.route('/')
def index():
    """Audit page — model selector, test categories, results."""
    categories = audit.list_categories()
    past_results = audit.list_results()

    # Get available models for the selector
    try:
        models = generate.list_all_network_models()
    except Exception:
        models = []

    return render_template('workshop/audit.html',
                           categories=categories,
                           past_results=past_results,
                           models=models)


# ---------------------------------------------------------------------------
# API — run audit
# ---------------------------------------------------------------------------

@audit_bp.route('/api/run', methods=['POST'])
def api_run():
    """Start a new audit run. Returns audit_id for SSE streaming."""
    data = request.json or {}
    model = data.get('model', '')
    host = data.get('host', '')
    categories = data.get('categories', [])
    system_prompt = data.get('system_prompt', '')

    if not model:
        return jsonify({'error': 'model required'}), 400

    audit_id = str(uuid.uuid4())[:12]

    # Create SSE queue for this audit
    q = queue.Queue(maxsize=200)
    with _audit_lock:
        _active_audits[audit_id] = {
            'queue': q,
            'model': model,
            'host': host,
            'done': False,
            'results': [],
        }

    # Run audit in background thread
    threading.Thread(
        target=_run_audit_thread,
        args=(audit_id, model, host, categories, system_prompt),
        daemon=True,
    ).start()

    return jsonify({'audit_id': audit_id})


def _run_audit_thread(audit_id, model, host, categories, system_prompt):
    """Background thread that runs the audit and pushes results to the SSE queue."""
    with _audit_lock:
        state = _active_audits.get(audit_id)
    if not state:
        return

    results = []
    try:
        for result in audit.run_audit(model, host=host or None,
                                      categories=categories or None,
                                      system_prompt=system_prompt or None):
            results.append(result)
            state['queue'].put(json.dumps({
                'type': 'result',
                **result,
            }))
    except Exception as e:
        state['queue'].put(json.dumps({
            'type': 'error',
            'error': str(e),
        }))

    # Save results
    if results:
        audit.save_results(audit_id, results, model, host)

    # Signal completion
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    state['queue'].put(json.dumps({
        'type': 'done',
        'total': total,
        'passed': passed,
        'failed': total - passed,
    }))

    with _audit_lock:
        state['done'] = True
        state['results'] = results


@audit_bp.route('/api/stream/<audit_id>')
def api_stream(audit_id):
    """SSE endpoint for live audit results."""
    with _audit_lock:
        state = _active_audits.get(audit_id)
    if not state:
        return jsonify({'error': 'Audit not found'}), 404

    def generate_events():
        try:
            while True:
                try:
                    msg = state['queue'].get(timeout=30)
                    yield f"data: {msg}\n\n"
                    data = json.loads(msg)
                    if data.get('type') == 'done':
                        return
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    if state.get('done'):
                        return
        finally:
            # Clean up after client disconnects
            with _audit_lock:
                if audit_id in _active_audits and state.get('done'):
                    del _active_audits[audit_id]

    return Response(stream_with_context(generate_events()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


# ---------------------------------------------------------------------------
# API — past results
# ---------------------------------------------------------------------------

@audit_bp.route('/api/results')
def api_results():
    """List past audit results."""
    return jsonify(audit.list_results())


@audit_bp.route('/api/results/<audit_id>')
def api_result_detail(audit_id):
    """Get full details of a past audit."""
    result = audit.get_result(audit_id)
    if not result:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(result)


@audit_bp.route('/api/tests')
def api_tests():
    """List available test categories."""
    return jsonify(audit.list_categories())
