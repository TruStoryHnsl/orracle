"""Pipeline blueprint — data processing, preview, rules, remote dispatch, video monitoring."""

from __future__ import annotations

import json
import os
import re
import time

from flask import Blueprint, Response, jsonify, render_template, request

import yaml

from nodes.base import NodeRegistry
from executor.dag import Pipeline
from executor.runner import PipelineRunner
from executor.preview import PreviewCache, generate_diff, compute_stats
from executor import remote as remote_executor

pipeline_bp = Blueprint('pipeline', __name__, url_prefix='/workshop/pipeline')

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
PIPELINES_DIR = os.path.join(CONFIG_DIR, 'pipelines')
RULES_DIR = os.path.join(CONFIG_DIR, 'rules')
PRESETS_DIR = os.path.join(CONFIG_DIR, 'presets')

runner = PipelineRunner()
preview_cache = PreviewCache(ttl=60)

# Manifest line count cache
_manifest_line_count = None


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@pipeline_bp.route('/')
def index():
    return render_template('pipeline/index.html')


@pipeline_bp.route('/editor')
def editor_page():
    return render_template('pipeline/editor.html')


@pipeline_bp.route('/rules')
def rules_page():
    return render_template('pipeline/rules.html')


@pipeline_bp.route('/pose-audit')
def pose_audit_page():
    return render_template('pipeline/pose_audit.html')


@pipeline_bp.route('/active')
def active_page():
    return render_template('pipeline/active.html')


# ---------------------------------------------------------------------------
# Rules API
# ---------------------------------------------------------------------------

@pipeline_bp.route('/api/rules/libraries')
def api_list_rule_libraries():
    from nodes.text.regex_rules import list_rule_libraries
    return jsonify(list_rule_libraries())


@pipeline_bp.route('/api/rules/library/<name>', methods=['GET'])
def api_get_rule_library(name):
    from nodes.text.regex_rules import load_rule_library
    rules = load_rule_library(name)
    return jsonify({'name': name, 'rules': rules})


@pipeline_bp.route('/api/rules/library/<name>', methods=['PUT'])
def api_save_rule_library(name):
    data = request.get_json() or {}
    rules = data.get('rules', [])
    os.makedirs(RULES_DIR, exist_ok=True)
    path = os.path.join(RULES_DIR, f'{name}.yaml')
    with open(path, 'w') as f:
        yaml.dump({'rules': rules}, f, default_flow_style=False,
                  sort_keys=False, allow_unicode=True)
    return jsonify({'saved': True, 'rule_count': len(rules)})


@pipeline_bp.route('/api/rules/test', methods=['POST'])
def api_test_rule():
    data = request.get_json() or {}
    pattern = data.get('pattern', '')
    sample = data.get('sample_text', '')
    try:
        compiled = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        return jsonify({'error': str(e), 'matches': []})
    matches = []
    for m in compiled.finditer(sample):
        matches.append({
            'start': m.start(), 'end': m.end(),
            'text': m.group()[:200],
            'line': sample[:m.start()].count('\n') + 1,
        })
    return jsonify({'matches': matches, 'count': len(matches)})


# ---------------------------------------------------------------------------
# Node types API
# ---------------------------------------------------------------------------

@pipeline_bp.route('/api/nodes/types')
def api_node_types():
    types = NodeRegistry.type_list()
    categories = {}
    for t in types:
        cat = t['category']
        categories.setdefault(cat, []).append(t)
    return jsonify({'types': types, 'categories': categories})


# ---------------------------------------------------------------------------
# Source scanning API
# ---------------------------------------------------------------------------

@pipeline_bp.route('/api/source/scan', methods=['POST'])
def api_scan_source():
    data = request.get_json() or {}
    directory = os.path.expanduser(data.get('directory', ''))
    if not directory:
        return jsonify({'error': 'No directory specified'})
    if not os.path.exists(directory):
        return jsonify({'error': f'Directory not found: {directory}'})

    skip_ext = {'.jpg', '.png', '.gif', '.pdf', '.css', '.js',
                '.mp3', '.mp4', '.zip', '.gz', '.ds_store'}
    entries = []
    categories = []

    try:
        top_names = sorted(os.listdir(directory))
    except OSError as e:
        return jsonify({'error': str(e)})

    for cat_name in top_names:
        if cat_name.startswith('.'):
            continue
        cat_path = os.path.join(directory, cat_name)
        try:
            names = os.listdir(cat_path)
        except (OSError, NotADirectoryError):
            continue
        count = 0
        for name in names:
            if name.startswith('.'):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in skip_ext:
                continue
            entries.append({
                'path': os.path.join(cat_path, name),
                'name': name,
                'category': cat_name,
            })
            count += 1
        if count > 0:
            categories.append({'name': cat_name, 'count': count})

    manifest_path = os.path.join(CONFIG_DIR, 'source_manifest.jsonl')
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    global _manifest_line_count
    _manifest_line_count = len(entries)

    return jsonify({
        'total_entries': len(entries),
        'categories': categories,
        'manifest': manifest_path,
    })


# ---------------------------------------------------------------------------
# Preview API
# ---------------------------------------------------------------------------

@pipeline_bp.route('/api/preview/sample', methods=['POST'])
def api_preview_sample():
    import random as rnd
    from nodes.text.metadata import extract_metadata
    from nodes.text.html_strip import strip_html, _is_html
    from nodes.text.header_strip import strip_email_headers
    from nodes.text.boilerplate import BoilerplateNode
    from nodes.text.regex_rules import load_rule_library, apply_rules
    from nodes.text.reflow import reflow_text
    from nodes.text.quality_filter import QualityFilterNode
    from nodes.base import DataChunk
    import unicodedata

    data = request.get_json() or {}
    directory = os.path.expanduser(data.get('directory', ''))
    use_random = data.get('random', False)
    idx = data.get('index', 0)

    manifest_path = os.path.join(CONFIG_DIR, 'source_manifest.jsonl')
    if not os.path.exists(manifest_path):
        return jsonify({'error': 'Click Scan first to build the story index'})

    global _manifest_line_count
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            if _manifest_line_count is None:
                _manifest_line_count = sum(1 for _ in f)
                f.seek(0)
            if _manifest_line_count == 0:
                return jsonify({'error': 'Manifest is empty'})
            target = rnd.randint(0, _manifest_line_count - 1) if use_random else idx % _manifest_line_count
            f.seek(0)
            for i, line in enumerate(f):
                if i == target:
                    entry = json.loads(line)
                    break
            else:
                return jsonify({'error': 'Could not read entry'})
    except (OSError, json.JSONDecodeError) as e:
        return jsonify({'error': str(e)})

    # Read the story
    path = entry['path']
    text = None
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        if raw and b'\x00' not in raw[:512]:
            text = raw.decode('utf-8', errors='replace')
    except IsADirectoryError:
        try:
            chapters = []
            for name in sorted(os.listdir(path)):
                if name.startswith('.'):
                    continue
                fp = os.path.join(path, name)
                try:
                    with open(fp, 'rb') as f:
                        raw = f.read()
                    if raw and b'\x00' not in raw[:512]:
                        chapters.append(raw.decode('utf-8', errors='replace'))
                except (OSError, IsADirectoryError):
                    continue
            if chapters:
                text = '\n\n---\n\n'.join(chapters)
        except OSError:
            pass
    except OSError:
        pass

    if text is None:
        return jsonify({'error': 'Could not read story'})

    # Run through pipeline stages
    stages = []
    original_text = text

    before = text
    if _is_html(text):
        text = strip_html(text)
    stages.append({'label': 'HTML Strip', 'before_label': 'Original',
                   'text_before': before, 'text_after': text,
                   'chars_before': len(before), 'chars_after': len(text),
                   'changed': before != text})

    before = text
    text = strip_email_headers(text)
    stages.append({'label': 'Email Headers', 'before_label': 'After HTML Strip',
                   'text_before': before, 'text_after': text,
                   'chars_before': len(before), 'chars_after': len(text),
                   'changed': before != text})

    before = text
    bp = BoilerplateNode()
    bp_chunk = DataChunk(text=text, metadata=entry)
    bp_result = bp.process({'text': [bp_chunk]}, {'use_defaults': True, 'patterns': '', 'scope_lines': 0})
    text = bp_result['cleaned'][0].text
    stages.append({'label': 'Boilerplate', 'before_label': 'After Headers',
                   'text_before': before, 'text_after': text,
                   'chars_before': len(before), 'chars_after': len(text),
                   'changed': before != text})

    before = text
    rules = load_rule_library('nifty_archive')
    text, match_count = apply_rules(text, rules)
    stages.append({'label': f'Rules ({match_count} matches)', 'before_label': 'After Boilerplate',
                   'text_before': before, 'text_after': text,
                   'chars_before': len(before), 'chars_after': len(text),
                   'changed': before != text})

    before = text
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = unicodedata.normalize('NFC', text)
    stages.append({'label': 'Normalize', 'before_label': 'After Rules',
                   'text_before': before, 'text_after': text,
                   'chars_before': len(before), 'chars_after': len(text),
                   'changed': before != text})

    before = text
    text = reflow_text(text, 80)
    stages.append({'label': 'Reflow', 'before_label': 'After Normalize',
                   'text_before': before, 'text_after': text,
                   'chars_before': len(before), 'chars_after': len(text),
                   'changed': before != text})

    qf = QualityFilterNode()
    qf_result = qf.process({'text': [DataChunk(text=text, metadata=entry)]}, {
        'min_chars': 500, 'max_chars': 2_000_000, 'min_words': 50,
        'max_non_ascii': 0.20, 'min_sentences': 3, 'max_avg_word_len': 15.0,
    })

    return jsonify({
        'name': entry['name'], 'category': entry['category'], 'path': entry['path'],
        'original_chars': len(original_text), 'final_chars': len(text),
        'quality_passed': len(qf_result['passed']) > 0, 'stages': stages,
    })


# ---------------------------------------------------------------------------
# Pose Audit API
# ---------------------------------------------------------------------------

@pipeline_bp.route('/api/pose/random')
def api_pose_random():
    import random as rnd
    frames_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'video_training', 'frames')
    if not os.path.isdir(frames_dir):
        return jsonify({'error': 'No frames directory'})
    jpgs = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    if not jpgs:
        return jsonify({'error': 'No frames'})
    return _pose_for_frame(os.path.join(frames_dir, rnd.choice(jpgs)), rnd.choice(jpgs))


@pipeline_bp.route('/api/pose/frame')
def api_pose_frame():
    idx = request.args.get('index', 0, type=int)
    frames_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'video_training', 'frames')
    if not os.path.isdir(frames_dir):
        return jsonify({'error': 'No frames'})
    jpgs = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not jpgs:
        return jsonify({'error': 'No frames'})
    name = jpgs[idx % len(jpgs)]
    return _pose_for_frame(os.path.join(frames_dir, name), name)


def _pose_for_frame(image_path, filename):
    from nodes.video.pose_extractor import _detect_pose_in_image, _compute_pose_features, _HAS_YOLO
    label = 'key_moment' if '_km.' in filename else 'not_key_moment'
    if _HAS_YOLO:
        pose = _detect_pose_in_image(image_path)
        if pose:
            return jsonify({
                'filename': filename, 'image_path': image_path, 'label': label,
                'landmarks': pose['landmarks'], 'features': _compute_pose_features(pose['landmarks']),
                'bbox': pose.get('bbox', []), 'confidence': pose.get('confidence', 0),
                'visible_landmarks': pose.get('visible_landmarks', 0), 'mediapipe_available': True,
            })
    return jsonify({
        'filename': filename, 'image_path': image_path, 'label': label,
        'landmarks': [], 'features': {'confidence': 0, 'has_person': False, 'magnitude': 0},
        'bbox': [], 'confidence': 0, 'visible_landmarks': 0, 'mediapipe_available': _HAS_YOLO if '_HAS_YOLO' in dir() else False,
    })


@pipeline_bp.route('/api/pose/image')
def api_pose_image():
    from flask import send_file
    path = request.args.get('path', '')
    if not path or not os.path.isfile(path):
        return 'Not found', 404
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    if not os.path.abspath(path).startswith(os.path.abspath(output_dir)):
        return 'Forbidden', 403
    return send_file(path, mimetype='image/jpeg')


# ---------------------------------------------------------------------------
# Video Pipeline Monitoring API
# ---------------------------------------------------------------------------

_VP_MACHINE = 'orrgate'

@pipeline_bp.route('/api/video/status')
def api_video_status():
    machine = remote_executor.get_machine(_VP_MACHINE)
    if not machine:
        return jsonify({'error': f'Machine {_VP_MACHINE} not configured'})
    target = f"{machine['user']}@{machine['host']}"
    r = remote_executor._ssh(target,
        'systemctl --user status orracle-video --no-pager 2>&1 | head -15', timeout=10)
    lines = r.get('stdout', '').splitlines()
    status_info = {'raw': r.get('stdout', ''), 'active': False}
    for line in lines:
        line = line.strip()
        if 'Active:' in line:
            status_info['active'] = 'active (running)' in line
            status_info['active_line'] = line
        if 'Memory:' in line:
            status_info['memory'] = line.split('Memory:')[1].strip()
    return jsonify(status_info)


@pipeline_bp.route('/api/video/log')
def api_video_log():
    n = request.args.get('n', 50, type=int)
    machine = remote_executor.get_machine(_VP_MACHINE)
    if not machine:
        return jsonify({'error': f'Machine {_VP_MACHINE} not configured'})
    target = f"{machine['user']}@{machine['host']}"
    r = remote_executor._ssh(target,
        f'journalctl --user -u orracle-video --no-pager -n {n} --output=short 2>/dev/null', timeout=15)
    lines = [l for l in r.get('stdout', '').splitlines() if l.strip()]
    return jsonify({'lines': lines, 'count': len(lines)})
