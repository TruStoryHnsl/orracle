"""LoRA Forge — Evolutionary model refinement engine.

Manages projects with tree-structured development paths:
- Sweep/Refine/Polish passes explore LoRA weight space
- Ternary feedback (keep/trash/neutral) narrows search regions
- Promoted LoRAs get baked into new checkpoint nodes
- Branching paths compete in tournament comparisons

Projects persist to config/projects/{id}.yaml.
Generation history logged to config/projects/{id}_history.csv.
"""

import csv
import os
import random
import threading
import time

import yaml

from . import comfyui

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
PROJECTS_DIR = os.path.join(CONFIG_DIR, 'projects')

_project_lock = threading.Lock()

# Quality tiers: (steps, resolution, candidate_count)
TIERS = {
    'sweep':  {'steps': 6,  'size': 512,  'candidates': 10},
    'refine': {'steps': 12, 'size': 768,  'candidates': 8},
    'polish': {'steps': 25, 'size': 1024, 'candidates': 5},
}


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

def _project_path(project_id: str) -> str:
    return os.path.join(PROJECTS_DIR, f'{project_id}.yaml')


def create_project(name: str, checkpoint: str, loras: list,
                   defaults: dict = None) -> dict:
    """Create a new forge project.

    Args:
        name: Human-readable project name
        checkpoint: Base checkpoint filename
        loras: List of LoRA names (strings)
        defaults: Optional default generation params
    """
    project_id = f'forge_{int(time.time())}_{random.randint(100, 999)}'
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    active_loras = [{'name': l} for l in loras]

    project = {
        'id': project_id,
        'name': name,
        'created': now,
        'updated': now,
        'defaults': {
            'prompt': '',
            'negative_prompt': '',
            'sampler': 'euler',
            'scheduler': 'normal',
            'cfg': 7.0,
            'seed': None,
            **(defaults or {}),
        },
        'tree': {
            'root': {
                'checkpoint': checkpoint,
                'active_loras': active_loras,
                'children': {},
            },
        },
        'current_node': 'root',
        'weight_regions': {},
        'passes': [],
        'tournaments': [],
    }

    # Initialize weight regions for root node
    _init_weight_regions(project, 'root', loras)

    save_project(project)
    return project


def _init_weight_regions(project: dict, node_path: str, lora_names: list):
    """Initialize 2D weight search regions for each LoRA at a node."""
    if node_path not in project['weight_regions']:
        project['weight_regions'][node_path] = {}

    for name in lora_names:
        if name not in project['weight_regions'][node_path]:
            project['weight_regions'][node_path][name] = {
                'model_min': 0.0,
                'model_max': 1.2,
                'clip_min': 0.0,
                'clip_max': 1.2,
                'scores': [],
            }


def load_project(project_id: str) -> dict | None:
    path = _project_path(project_id)
    try:
        with _project_lock:
            with open(path) as f:
                return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        return None


def save_project(project: dict):
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    project['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    path = _project_path(project['id'])
    with _project_lock:
        with open(path, 'w') as f:
            yaml.dump(project, f, default_flow_style=False, sort_keys=False)


def list_projects() -> list:
    """List all projects (summary only)."""
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    projects = []
    for fname in os.listdir(PROJECTS_DIR):
        if not fname.endswith('.yaml'):
            continue
        try:
            with open(os.path.join(PROJECTS_DIR, fname)) as f:
                p = yaml.safe_load(f)
            if p and 'id' in p:
                projects.append({
                    'id': p['id'],
                    'name': p.get('name', ''),
                    'created': p.get('created', ''),
                    'updated': p.get('updated', ''),
                    'current_node': p.get('current_node', 'root'),
                    'pass_count': len(p.get('passes', [])),
                    'checkpoint': _get_root_checkpoint(p),
                })
        except (yaml.YAMLError, OSError):
            continue
    projects.sort(key=lambda p: p['updated'], reverse=True)
    return projects


def delete_project(project_id: str) -> bool:
    path = _project_path(project_id)
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


def _get_root_checkpoint(project: dict) -> str:
    tree = project.get('tree', {})
    root = tree.get('root', {})
    return root.get('checkpoint', '')


# ---------------------------------------------------------------------------
# Tree navigation
# ---------------------------------------------------------------------------

def get_node(project: dict, path: str) -> dict | None:
    """Traverse dot-separated path to find a tree node.

    'root' → tree['root']
    'root.branch_a' → tree['root']['children']['branch_a']
    """
    parts = path.split('.')
    if not parts or parts[0] != 'root':
        return None

    node = project['tree'].get('root')
    if not node:
        return None

    for part in parts[1:]:
        children = node.get('children', {})
        node = children.get(part)
        if node is None:
            return None

    return node


def navigate(project: dict, path: str) -> bool:
    """Set the current working node."""
    node = get_node(project, path)
    if node is None:
        return False
    project['current_node'] = path
    return True


def get_tree_summary(project: dict) -> dict:
    """Build a serializable tree summary for the UI."""
    def _summarize(node, path):
        summary = {
            'path': path,
            'checkpoint': node.get('checkpoint', ''),
            'lora_count': len(node.get('active_loras', [])),
            'children': {},
        }
        if 'promoted_lora' in node:
            summary['promoted_lora'] = node['promoted_lora']
            summary['promoted_model_weight'] = node.get('promoted_model_weight', 0)
            summary['promoted_clip_weight'] = node.get('promoted_clip_weight', 0)
        for name, child in node.get('children', {}).items():
            summary['children'][name] = _summarize(child, f'{path}.{name}')
        return summary

    root = project['tree'].get('root')
    if not root:
        return {}
    return _summarize(root, 'root')


# ---------------------------------------------------------------------------
# Pass generation — weight exploration
# ---------------------------------------------------------------------------

def start_pass(project: dict, tier: str, lora_focus: str,
               comfyui_url: str = None) -> dict | None:
    """Generate a comparison pass for a specific LoRA at the current node.

    Args:
        project: Full project dict
        tier: 'sweep', 'refine', or 'polish'
        lora_focus: LoRA name to explore weights for
        comfyui_url: ComfyUI API URL

    Returns pass info dict or None on error.
    """
    if tier not in TIERS:
        return None

    url = comfyui_url or comfyui.COMFYUI_DEFAULT
    node_path = project['current_node']
    node = get_node(project, node_path)
    if not node:
        return None

    # Verify lora_focus is active at this node
    active_names = [l['name'] for l in node.get('active_loras', [])]
    if lora_focus not in active_names:
        return None

    # Get or init weight region
    _init_weight_regions(project, node_path, active_names)
    region = project['weight_regions'][node_path].get(lora_focus)
    if not region:
        return None

    tier_cfg = TIERS[tier]
    num_candidates = tier_cfg['candidates']
    size = tier_cfg['size']
    steps = tier_cfg['steps']

    # Generate candidate weight pairs (model_w, clip_w)
    candidates = _generate_candidates(region, tier, num_candidates)

    pass_num = len(project['passes']) + 1
    seed = project['defaults'].get('seed') or random.randint(0, 2**32 - 1)

    pass_data = {
        'pass_num': pass_num,
        'tier': tier,
        'node_path': node_path,
        'lora_focus': lora_focus,
        'seed': seed,
        'status': 'generating',
        'candidates': candidates,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    project['passes'].append(pass_data)
    save_project(project)

    # Submit generation in background
    gen_config = {
        'url': url,
        'project_id': project['id'],
        'pass_num': pass_num,
        'checkpoint': node['checkpoint'],
        'active_loras': node.get('active_loras', []),
        'lora_focus': lora_focus,
        'candidates': candidates,
        'prompt': project['defaults'].get('prompt', ''),
        'negative_prompt': project['defaults'].get('negative_prompt', ''),
        'seed': seed,
        'steps': steps,
        'size': size,
        'cfg': project['defaults'].get('cfg', 7.0),
        'sampler': project['defaults'].get('sampler', 'euler'),
        'scheduler': project['defaults'].get('scheduler', 'normal'),
    }

    threading.Thread(target=_run_pass, args=(gen_config,), daemon=True).start()

    return {
        'pass_num': pass_num,
        'tier': tier,
        'candidate_count': len(candidates),
        'status': 'generating',
    }


def _generate_candidates(region: dict, tier: str, count: int) -> list:
    """Generate candidate weight pairs based on tier and current region."""
    candidates = []

    m_min, m_max = region['model_min'], region['model_max']
    c_min, c_max = region['clip_min'], region['clip_max']

    # Get keeps from existing scores for clustering
    keeps = [s for s in region.get('scores', []) if s.get('label') == 'keep']

    if tier == 'sweep':
        # Uniform grid across remaining range
        # Use a grid that fills count candidates
        import math
        grid_side = int(math.ceil(math.sqrt(count)))
        m_steps = max(grid_side, 2)
        c_steps = max(count // m_steps, 2)

        for i in range(m_steps):
            for j in range(c_steps):
                if len(candidates) >= count:
                    break
                mw = m_min + (m_max - m_min) * i / max(m_steps - 1, 1)
                cw = c_min + (c_max - c_min) * j / max(c_steps - 1, 1)
                candidates.append(_make_candidate(len(candidates), mw, cw))
            if len(candidates) >= count:
                break

    elif tier == 'refine':
        # Gaussian samples clustered around keeps
        if keeps:
            mean_m = sum(s['model_w'] for s in keeps) / len(keeps)
            mean_c = sum(s['clip_w'] for s in keeps) / len(keeps)
            spread_m = (m_max - m_min) * 0.2
            spread_c = (c_max - c_min) * 0.2
        else:
            mean_m = (m_min + m_max) / 2
            mean_c = (c_min + c_max) / 2
            spread_m = (m_max - m_min) * 0.3
            spread_c = (c_max - c_min) * 0.3

        for i in range(count):
            mw = _clamp(random.gauss(mean_m, spread_m), m_min, m_max)
            cw = _clamp(random.gauss(mean_c, spread_c), c_min, c_max)
            candidates.append(_make_candidate(i, mw, cw))

    elif tier == 'polish':
        # Very tight variations around best keeps
        if keeps:
            # Sort by recency, weight toward recent keeps
            best = keeps[-1]
            spread_m = (m_max - m_min) * 0.08
            spread_c = (c_max - c_min) * 0.08
            base_m, base_c = best['model_w'], best['clip_w']
        else:
            base_m = (m_min + m_max) / 2
            base_c = (c_min + c_max) / 2
            spread_m = (m_max - m_min) * 0.12
            spread_c = (c_max - c_min) * 0.12

        for i in range(count):
            mw = _clamp(random.gauss(base_m, spread_m), m_min, m_max)
            cw = _clamp(random.gauss(base_c, spread_c), c_min, c_max)
            candidates.append(_make_candidate(i, mw, cw))

    return candidates


def _make_candidate(index: int, model_w: float, clip_w: float) -> dict:
    return {
        'id': f'c{index}',
        'model_w': round(model_w, 3),
        'clip_w': round(clip_w, 3),
        'prompt_id': None,
        'status': 'queued',
        'image_filename': None,
        'image_subfolder': None,
        'label': None,  # 'keep', 'trash', or None (neutral)
    }


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _run_pass(config: dict):
    """Background thread: submit all candidates to ComfyUI and poll results."""
    url = config['url']
    project_id = config['project_id']
    pass_num = config['pass_num']

    for candidate in config['candidates']:
        # Build LoRA list: focus LoRA uses candidate weights, others use midpoint
        loras = []
        for lora_info in config['active_loras']:
            name = lora_info['name']
            if name == config['lora_focus']:
                loras.append({
                    'name': name,
                    'model_strength': candidate['model_w'],
                    'clip_strength': candidate['clip_w'],
                })
            else:
                # Non-focus LoRAs at 0.5/0.5 default
                loras.append({
                    'name': name,
                    'model_strength': 0.5,
                    'clip_strength': 0.5,
                })

        workflow = comfyui.build_workflow(
            checkpoint=config['checkpoint'],
            loras=loras,
            prompt=config['prompt'],
            negative_prompt=config['negative_prompt'],
            seed=config['seed'],
            steps=config['steps'],
            cfg=config['cfg'],
            width=config['size'],
            height=config['size'],
            sampler=config['sampler'],
            scheduler=config['scheduler'],
        )

        prompt_id = comfyui.queue_prompt(url, workflow)
        if prompt_id:
            candidate['prompt_id'] = prompt_id
            candidate['status'] = 'generating'
        else:
            candidate['status'] = 'failed'

    # Poll for completion
    pending = [c for c in config['candidates'] if c['status'] == 'generating']
    deadline = time.time() + 600

    while pending and time.time() < deadline:
        time.sleep(2)
        still_pending = []
        for c in pending:
            entry = comfyui.poll_prompt_completion(url, c['prompt_id'], timeout=0)
            if entry:
                images = comfyui.extract_images(entry)
                if images:
                    c['image_filename'] = images[0]['filename']
                    c['image_subfolder'] = images[0].get('subfolder', '')
                    c['status'] = 'done'
                    # Claim output: download locally, delete from ComfyUI
                    img_bytes = comfyui.fetch_image(url, c['image_filename'],
                                                    c['image_subfolder'])
                    if img_bytes:
                        comfyui.save_output(c['image_filename'], img_bytes)
                    try:
                        comfyui.delete_files(url, [c['image_filename']])
                    except Exception:
                        pass  # best-effort cleanup
                else:
                    still_pending.append(c)
            else:
                still_pending.append(c)
        pending = still_pending

    for c in pending:
        c['status'] = 'failed'

    # Update project on disk
    project = load_project(project_id)
    if project and pass_num <= len(project['passes']):
        project['passes'][pass_num - 1]['candidates'] = config['candidates']
        project['passes'][pass_num - 1]['status'] = 'labeling'
        save_project(project)


# ---------------------------------------------------------------------------
# Ternary feedback
# ---------------------------------------------------------------------------

def submit_feedback(project: dict, pass_num: int, labels: dict) -> bool:
    """Apply ternary labels to candidates and update weight regions.

    Args:
        project: Full project dict
        pass_num: 1-based pass number
        labels: {candidate_id: 'keep'|'trash'|None}
    """
    if pass_num < 1 or pass_num > len(project['passes']):
        return False

    pass_data = project['passes'][pass_num - 1]
    if pass_data['status'] != 'labeling':
        return False

    node_path = pass_data['node_path']
    lora_focus = pass_data['lora_focus']

    # Apply labels to candidates
    for c in pass_data['candidates']:
        if c['id'] in labels:
            c['label'] = labels[c['id']]

    # Update weight regions based on feedback
    region = project['weight_regions'].get(node_path, {}).get(lora_focus)
    if region:
        _apply_feedback(region, pass_data['candidates'])

    pass_data['status'] = 'done'
    save_project(project)

    # Log to CSV history for persistent parameter tracking
    log_pass_to_history(project, pass_num)

    return True


def _apply_feedback(region: dict, candidates: list):
    """Update weight region boundaries based on ternary feedback.

    - Trash: shrink boundaries past trashed values
    - Keep: record as scored point for clustering
    - Neutral: no region change, but record for density tracking
    """
    trashed = [c for c in candidates if c.get('label') == 'trash']
    kept = [c for c in candidates if c.get('label') == 'keep']

    # Record all labeled scores
    for c in candidates:
        if c.get('label') in ('keep', 'trash'):
            region['scores'].append({
                'model_w': c['model_w'],
                'clip_w': c['clip_w'],
                'label': c['label'],
            })

    if not trashed and not kept:
        return

    # Shrink region based on trashed candidates
    # If trashed candidates are at extremes, pull boundaries inward
    if trashed:
        trash_m = [c['model_w'] for c in trashed]
        trash_c = [c['clip_w'] for c in trashed]

        # Only shrink if trashed values are at the edge of the region
        m_center = (region['model_min'] + region['model_max']) / 2
        c_center = (region['clip_min'] + region['clip_max']) / 2

        for mw in trash_m:
            # If trashed weight is below center, raise the min
            if mw < m_center:
                region['model_min'] = max(region['model_min'], mw + 0.02)
            else:
                region['model_max'] = min(region['model_max'], mw - 0.02)

        for cw in trash_c:
            if cw < c_center:
                region['clip_min'] = max(region['clip_min'], cw + 0.02)
            else:
                region['clip_max'] = min(region['clip_max'], cw - 0.02)

        # Ensure min < max with minimum span
        if region['model_max'] - region['model_min'] < 0.05:
            mid = (region['model_min'] + region['model_max']) / 2
            region['model_min'] = max(0.0, mid - 0.025)
            region['model_max'] = min(1.5, mid + 0.025)
        if region['clip_max'] - region['clip_min'] < 0.05:
            mid = (region['clip_min'] + region['clip_max']) / 2
            region['clip_min'] = max(0.0, mid - 0.025)
            region['clip_max'] = min(1.5, mid + 0.025)


# ---------------------------------------------------------------------------
# LoRA promotion — bake into checkpoint
# ---------------------------------------------------------------------------

def promote_lora(project: dict, lora_name: str,
                 model_weight: float, clip_weight: float,
                 branch_name: str, comfyui_url: str = None) -> dict:
    """Bake a LoRA into the current checkpoint, creating a new branch node.

    Returns status dict. The bake runs asynchronously.
    """
    url = comfyui_url or comfyui.COMFYUI_DEFAULT
    node_path = project['current_node']
    node = get_node(project, node_path)
    if not node:
        return {'error': 'Current node not found'}

    # Verify LoRA is active at this node
    active_names = [l['name'] for l in node.get('active_loras', [])]
    if lora_name not in active_names:
        return {'error': f'{lora_name} not active at {node_path}'}

    # Sanitize branch name
    branch_name = branch_name.strip().replace(' ', '_').replace('.', '_')
    if not branch_name:
        branch_name = f'br_{int(time.time()) % 10000}'

    if branch_name in node.get('children', {}):
        return {'error': f'Branch {branch_name} already exists'}

    # Create child node with remaining LoRAs
    remaining_loras = [l for l in node['active_loras'] if l['name'] != lora_name]
    output_prefix = f'forge_{project["id"]}_{branch_name}'

    child = {
        'promoted_lora': lora_name,
        'promoted_model_weight': model_weight,
        'promoted_clip_weight': clip_weight,
        'checkpoint': f'{output_prefix}_00001_.safetensors',
        'bake_status': 'pending',
        'active_loras': remaining_loras,
        'children': {},
    }

    if 'children' not in node:
        node['children'] = {}
    node['children'][branch_name] = child

    # Init weight regions for remaining LoRAs at new node
    new_path = f'{node_path}.{branch_name}'
    _init_weight_regions(project, new_path, [l['name'] for l in remaining_loras])

    save_project(project)

    # Start bake in background
    bake_config = {
        'url': url,
        'project_id': project['id'],
        'node_path': node_path,
        'branch_name': branch_name,
        'checkpoint': node['checkpoint'],
        'lora': lora_name,
        'model_weight': model_weight,
        'clip_weight': clip_weight,
        'output_prefix': output_prefix,
    }
    threading.Thread(target=_run_bake, args=(bake_config,), daemon=True).start()

    return {
        'ok': True,
        'branch': branch_name,
        'new_path': new_path,
        'status': 'baking',
    }


def _run_bake(config: dict):
    """Background: submit bake workflow and wait for completion."""
    url = config['url']
    workflow = comfyui.build_bake_workflow(
        checkpoint=config['checkpoint'],
        lora=config['lora'],
        model_weight=config['model_weight'],
        clip_weight=config['clip_weight'],
        output_prefix=config['output_prefix'],
    )

    prompt_id = comfyui.queue_prompt(url, workflow)
    status = 'failed'

    if prompt_id:
        entry = comfyui.poll_prompt_completion(url, prompt_id, timeout=1200)
        if entry:
            status = 'done'

    # Update project
    project = load_project(config['project_id'])
    if project:
        node = get_node(project, config['node_path'])
        if node:
            child = node.get('children', {}).get(config['branch_name'])
            if child:
                child['bake_status'] = status
                save_project(project)


# ---------------------------------------------------------------------------
# Branching (fork without promotion)
# ---------------------------------------------------------------------------

def create_branch(project: dict, name: str, from_path: str = None) -> dict:
    """Fork a development path from the current or specified node."""
    from_path = from_path or project['current_node']
    node = get_node(project, from_path)
    if not node:
        return {'error': 'Source node not found'}

    name = name.strip().replace(' ', '_').replace('.', '_')
    if not name:
        return {'error': 'Branch name required'}

    if name in node.get('children', {}):
        return {'error': f'Branch {name} already exists'}

    # Fork: same checkpoint, same LoRAs
    child = {
        'checkpoint': node['checkpoint'],
        'active_loras': list(node.get('active_loras', [])),
        'children': {},
    }

    if 'children' not in node:
        node['children'] = {}
    node['children'][name] = child

    new_path = f'{from_path}.{name}'
    lora_names = [l['name'] for l in child['active_loras']]
    _init_weight_regions(project, new_path, lora_names)

    save_project(project)
    return {'ok': True, 'branch': name, 'new_path': new_path}


# ---------------------------------------------------------------------------
# Tournaments — head-to-head branch comparison
# ---------------------------------------------------------------------------

def start_tournament(project: dict, branch_a: str, branch_b: str,
                     comfyui_url: str = None, num_seeds: int = 4) -> dict:
    """Start a head-to-head comparison between two branches.

    Generates images from each branch using the same seeds for fair comparison.
    """
    url = comfyui_url or comfyui.COMFYUI_DEFAULT

    node_a = get_node(project, branch_a)
    node_b = get_node(project, branch_b)
    if not node_a or not node_b:
        return {'error': 'One or both branches not found'}

    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]
    tournament_id = len(project.get('tournaments', [])) + 1

    tournament = {
        'id': tournament_id,
        'branch_a': branch_a,
        'branch_b': branch_b,
        'seeds': seeds,
        'status': 'generating',
        'pairs': [],
        'verdict': None,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Build pairs (one per seed)
    for i, seed in enumerate(seeds):
        tournament['pairs'].append({
            'seed': seed,
            'a': {'prompt_id': None, 'status': 'queued', 'image_filename': None, 'image_subfolder': None},
            'b': {'prompt_id': None, 'status': 'queued', 'image_filename': None, 'image_subfolder': None},
            'winner': None,
        })

    if 'tournaments' not in project:
        project['tournaments'] = []
    project['tournaments'].append(tournament)
    save_project(project)

    # Generate in background
    gen_config = {
        'url': url,
        'project_id': project['id'],
        'tournament_id': tournament_id,
        'node_a': node_a,
        'node_b': node_b,
        'branch_a': branch_a,
        'branch_b': branch_b,
        'defaults': project['defaults'],
        'seeds': seeds,
    }
    threading.Thread(target=_run_tournament, args=(gen_config,), daemon=True).start()

    return {
        'ok': True,
        'tournament_id': tournament_id,
        'seed_count': num_seeds,
        'status': 'generating',
    }


def _run_tournament(config: dict):
    """Background: generate tournament images for both branches."""
    url = config['url']
    defaults = config['defaults']

    for i, seed in enumerate(config['seeds']):
        for side, key in [('a', 'node_a'), ('b', 'node_b')]:
            node = config[key]
            loras = []
            for l in node.get('active_loras', []):
                loras.append({
                    'name': l['name'],
                    'model_strength': 0.5,
                    'clip_strength': 0.5,
                })

            workflow = comfyui.build_workflow(
                checkpoint=node['checkpoint'],
                loras=loras,
                prompt=defaults.get('prompt', ''),
                negative_prompt=defaults.get('negative_prompt', ''),
                seed=seed,
                steps=25,
                cfg=defaults.get('cfg', 7.0),
                width=1024, height=1024,
                sampler=defaults.get('sampler', 'euler'),
                scheduler=defaults.get('scheduler', 'normal'),
            )

            prompt_id = comfyui.queue_prompt(url, workflow)
            if prompt_id:
                _update_tournament_candidate(
                    config['project_id'], config['tournament_id'],
                    i, side, 'prompt_id', prompt_id)
                _update_tournament_candidate(
                    config['project_id'], config['tournament_id'],
                    i, side, 'status', 'generating')

                entry = comfyui.poll_prompt_completion(url, prompt_id)
                if entry:
                    images = comfyui.extract_images(entry)
                    if images:
                        _update_tournament_candidate(
                            config['project_id'], config['tournament_id'],
                            i, side, 'image_filename', images[0]['filename'])
                        _update_tournament_candidate(
                            config['project_id'], config['tournament_id'],
                            i, side, 'image_subfolder', images[0].get('subfolder', ''))
                        _update_tournament_candidate(
                            config['project_id'], config['tournament_id'],
                            i, side, 'status', 'done')
                        continue

            _update_tournament_candidate(
                config['project_id'], config['tournament_id'],
                i, side, 'status', 'failed')

    # Mark tournament as voting
    project = load_project(config['project_id'])
    if project:
        t = _get_tournament(project, config['tournament_id'])
        if t:
            t['status'] = 'voting'
            save_project(project)


def _update_tournament_candidate(project_id, tournament_id, pair_idx, side, key, value):
    """Thread-safe update of a single tournament candidate field."""
    project = load_project(project_id)
    if not project:
        return
    t = _get_tournament(project, tournament_id)
    if t and pair_idx < len(t['pairs']):
        t['pairs'][pair_idx][side][key] = value
        save_project(project)


def _get_tournament(project: dict, tournament_id: int) -> dict | None:
    for t in project.get('tournaments', []):
        if t['id'] == tournament_id:
            return t
    return None


def submit_tournament_vote(project: dict, tournament_id: int,
                           votes: dict) -> bool:
    """Submit votes for tournament pairs.

    Args:
        votes: {pair_index: 'a'|'b'|'tie'}
    """
    t = _get_tournament(project, tournament_id)
    if not t or t['status'] != 'voting':
        return False

    for idx_str, winner in votes.items():
        idx = int(idx_str)
        if idx < len(t['pairs']):
            t['pairs'][idx]['winner'] = winner

    # Tally
    a_wins = sum(1 for p in t['pairs'] if p.get('winner') == 'a')
    b_wins = sum(1 for p in t['pairs'] if p.get('winner') == 'b')

    if a_wins > b_wins:
        t['verdict'] = {'winner': 'a', 'branch': t['branch_a'], 'score': f'{a_wins}-{b_wins}'}
    elif b_wins > a_wins:
        t['verdict'] = {'winner': 'b', 'branch': t['branch_b'], 'score': f'{a_wins}-{b_wins}'}
    else:
        t['verdict'] = {'winner': 'tie', 'score': f'{a_wins}-{b_wins}'}

    t['status'] = 'done'
    save_project(project)
    return True


# ---------------------------------------------------------------------------
# Image proxy helper
# ---------------------------------------------------------------------------

def get_candidate_image(project: dict, pass_num: int,
                        candidate_id: str) -> tuple | None:
    """Find a candidate's image info. Returns (filename, subfolder) or None."""
    if pass_num < 1 or pass_num > len(project.get('passes', [])):
        return None

    pass_data = project['passes'][pass_num - 1]
    for c in pass_data.get('candidates', []):
        if c['id'] == candidate_id and c.get('image_filename'):
            return (c['image_filename'], c.get('image_subfolder', ''))
    return None


def get_tournament_image(project: dict, tournament_id: int,
                         pair_idx: int, side: str) -> tuple | None:
    """Find a tournament image. Returns (filename, subfolder) or None."""
    t = _get_tournament(project, tournament_id)
    if not t or pair_idx >= len(t['pairs']):
        return None
    entry = t['pairs'][pair_idx].get(side, {})
    if entry.get('image_filename'):
        return (entry['image_filename'], entry.get('image_subfolder', ''))
    return None


# ---------------------------------------------------------------------------
# Generation History CSV — persistent parameter log
# ---------------------------------------------------------------------------

_CSV_HEADERS = [
    'timestamp', 'project_id', 'pass_num', 'tier', 'candidate_id',
    'lora_focus', 'model_weight', 'clip_weight',
    'checkpoint', 'prompt', 'negative_prompt',
    'seed', 'steps', 'cfg', 'width', 'height',
    'sampler', 'scheduler', 'image_filename', 'label',
]


def _history_path(project_id: str) -> str:
    return os.path.join(PROJECTS_DIR, f'{project_id}_history.csv')


def _append_history(project_id: str, rows: list):
    """Append generation records to the project's CSV history.

    Each row is a dict matching _CSV_HEADERS keys.
    Creates the file with headers if it doesn't exist.
    """
    path = _history_path(project_id)
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def log_pass_to_history(project: dict, pass_num: int):
    """Write all candidates from a completed pass to the CSV history."""
    if pass_num < 1 or pass_num > len(project.get('passes', [])):
        return

    pass_data = project['passes'][pass_num - 1]
    node = get_node(project, pass_data.get('node_path', 'root'))
    defaults = project.get('defaults', {})
    tier_cfg = TIERS.get(pass_data.get('tier', 'sweep'), TIERS['sweep'])

    rows = []
    for c in pass_data.get('candidates', []):
        rows.append({
            'timestamp': pass_data.get('started', ''),
            'project_id': project['id'],
            'pass_num': pass_num,
            'tier': pass_data.get('tier', ''),
            'candidate_id': c['id'],
            'lora_focus': pass_data.get('lora_focus', ''),
            'model_weight': c.get('model_w', 0),
            'clip_weight': c.get('clip_w', 0),
            'checkpoint': node.get('checkpoint', '') if node else '',
            'prompt': defaults.get('prompt', ''),
            'negative_prompt': defaults.get('negative_prompt', ''),
            'seed': pass_data.get('seed', ''),
            'steps': tier_cfg['steps'],
            'cfg': defaults.get('cfg', 7.0),
            'width': tier_cfg['size'],
            'height': tier_cfg['size'],
            'sampler': defaults.get('sampler', ''),
            'scheduler': defaults.get('scheduler', ''),
            'image_filename': c.get('image_filename', ''),
            'label': c.get('label', ''),
        })

    _append_history(project['id'], rows)


def get_history_csv_path(project_id: str) -> str | None:
    """Return the CSV history path if it exists."""
    path = _history_path(project_id)
    return path if os.path.exists(path) else None


def load_history(project_id: str) -> list:
    """Load all history rows as dicts."""
    path = _history_path(project_id)
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))
