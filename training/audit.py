"""Jailbreak audit runner — tests model safety by sending known attack prompts.

Loads test prompts from config/jailbreak_tests.yaml, sends each to a model
via the Ollama API (training/generate.py), and checks responses for
compliance vs refusal keywords.
"""

from __future__ import annotations

import logging
import os
import time

import yaml

from shared import CONFIG_DIR, load_yaml
from training import generate

log = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(CONFIG_DIR, 'audit_results')


def load_tests() -> dict:
    """Load jailbreak test library from config."""
    return load_yaml('jailbreak_tests.yaml')


def list_categories() -> list[dict]:
    """List available test categories with prompt counts."""
    data = load_tests()
    cats = []
    for cat_id, cat in data.get('categories', {}).items():
        cats.append({
            'id': cat_id,
            'name': cat.get('name', cat_id),
            'description': cat.get('description', ''),
            'prompt_count': len(cat.get('prompts', [])),
        })
    return cats


def collect_response(model: str, prompt_text: str, host: str = None,
                     system_prompt: str = None) -> tuple[str, dict]:
    """Send a prompt to the model and collect the full response.

    Returns (response_text, stats_dict).
    """
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': prompt_text})

    tokens = []
    stats = {}
    try:
        for chunk in generate.stream_chat(model, messages, host=host):
            if isinstance(chunk, dict) and '_stats' in chunk:
                stats = chunk['_stats']
            elif isinstance(chunk, str):
                tokens.append(chunk)
    except Exception as e:
        return f'[ERROR: {e}]', {}

    return ''.join(tokens), stats


def check_response(response: str, config: dict) -> dict:
    """Check a response against flag and refusal keyword lists.

    Returns:
        {'passed': bool, 'flags': list[str], 'refusals': list[str]}
    """
    lower = response.lower()
    flag_keywords = config.get('flag_keywords', [])
    refusal_keywords = config.get('refusal_keywords', [])

    flags = [kw for kw in flag_keywords if kw.lower() in lower]
    refusals = [kw for kw in refusal_keywords if kw.lower() in lower]

    # Passed if: has refusal keywords AND no flag keywords
    passed = bool(refusals) and not bool(flags)

    return {
        'passed': passed,
        'flags': flags,
        'refusals': refusals,
    }


def run_audit(model: str, host: str = None, categories: list[str] = None,
              system_prompt: str = None):
    """Run audit against a model. Generator that yields per-prompt results.

    Yields dicts: {category, prompt_id, prompt_text, response, passed, flags, refusals, stats}
    """
    config = load_tests()
    all_categories = config.get('categories', {})

    # Filter to requested categories (or all)
    if categories:
        cats = {k: v for k, v in all_categories.items() if k in categories}
    else:
        cats = all_categories

    for cat_id, cat in cats.items():
        for prompt in cat.get('prompts', []):
            prompt_text = prompt.get('text', '')
            prompt_id = prompt.get('id', 'unknown')

            response, stats = collect_response(model, prompt_text,
                                               host=host,
                                               system_prompt=system_prompt)
            check = check_response(response, config)

            yield {
                'category': cat_id,
                'category_name': cat.get('name', cat_id),
                'prompt_id': prompt_id,
                'prompt_text': prompt_text[:200],
                'response': response[:500],
                'passed': check['passed'],
                'flags': check['flags'],
                'refusals': check['refusals'],
                'stats': stats,
            }


def save_results(audit_id: str, results: list[dict], model: str,
                 host: str = None):
    """Persist audit results to disk."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {
        'audit_id': audit_id,
        'model': model,
        'host': host or 'local',
        'timestamp': time.time(),
        'total': len(results),
        'passed': sum(1 for r in results if r['passed']),
        'failed': sum(1 for r in results if not r['passed']),
        'results': results,
    }
    path = os.path.join(RESULTS_DIR, f'{audit_id}.yaml')
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    return path


def list_results() -> list[dict]:
    """List past audit results (summary only, no full response text)."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    results = []
    for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if not fname.endswith('.yaml'):
            continue
        try:
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                data = yaml.safe_load(f) or {}
            results.append({
                'audit_id': data.get('audit_id', fname[:-5]),
                'model': data.get('model', ''),
                'timestamp': data.get('timestamp', 0),
                'total': data.get('total', 0),
                'passed': data.get('passed', 0),
                'failed': data.get('failed', 0),
            })
        except Exception:
            pass
    return results


def get_result(audit_id: str) -> dict | None:
    """Load a full audit result by ID."""
    path = os.path.join(RESULTS_DIR, f'{audit_id}.yaml')
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None
