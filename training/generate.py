from __future__ import annotations

"""Text generation via Ollama API.

Proxies streaming chat/completion requests to Ollama instances.
Supports local and remote Ollama endpoints (e.g. orrion via Tailscale).
No external dependencies — uses stdlib urllib.
"""

import json
import os
import urllib.request
import urllib.error

import yaml

OLLAMA_DEFAULT = 'http://localhost:11434'
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')


def _get_ollama_base(host: str = None) -> str:
    """Resolve Ollama base URL.

    Priority: explicit host arg > OLLAMA_HOST env > model_registry default > localhost.
    """
    if host:
        # Allow bare hostname (add http:// and port if needed)
        if not host.startswith('http'):
            host = f'http://{host}'
        if host.count(':') < 2:  # no port specified
            host = f'{host}:11434'
        return host
    return os.environ.get('OLLAMA_HOST', OLLAMA_DEFAULT)


def ollama_available(host: str = None) -> bool:
    """Check if Ollama API is reachable."""
    base = _get_ollama_base(host)
    try:
        req = urllib.request.Request(f'{base}/api/tags')
        with urllib.request.urlopen(req, timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def list_models(host: str = None) -> list:
    """List models from Ollama API."""
    base = _get_ollama_base(host)
    try:
        req = urllib.request.Request(f'{base}/api/tags')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = []
            for m in data.get('models', []):
                details = m.get('details', {})
                models.append({
                    'name': m['name'],
                    'size': m.get('size', 0),
                    'size_gb': round(m.get('size', 0) / (1024**3), 1),
                    'family': details.get('family', ''),
                    'params': details.get('parameter_size', ''),
                    'quant': details.get('quantization_level', ''),
                    'host': base,
                })
            return models
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


def list_all_network_models() -> list:
    """List models from all registered Ollama hosts on the network."""
    all_models = []

    # Local first
    local = list_models()
    for m in local:
        m['source'] = 'local'
    all_models.extend(local)

    # Check registered machines for ollama_serve capability
    registry = load_model_registry()
    for host_entry in registry.get('ollama_hosts', []):
        host = host_entry.get('url', '')
        name = host_entry.get('name', host)
        if not host:
            continue
        try:
            remote_models = list_models(host)
            for m in remote_models:
                m['source'] = name
            all_models.extend(remote_models)
        except Exception:
            continue

    return all_models


def stream_chat(model: str, messages: list, options: dict = None,
                host: str = None):
    """Stream chat completion from Ollama. Yields content strings.

    Args:
        model: Ollama model name (e.g. 'mistral:latest')
        messages: List of {role, content} dicts
        options: Ollama options dict (temperature, top_p, etc.)
        host: Optional Ollama host URL (for remote instances)

    Yields:
        str: Token chunks as they arrive
    """
    base = _get_ollama_base(host)
    payload = json.dumps({
        'model': model,
        'messages': messages,
        'stream': True,
        'options': options or {},
    }).encode()

    req = urllib.request.Request(
        f'{base}/api/chat',
        data=payload,
        headers={'Content-Type': 'application/json'},
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = chunk.get('message', {}).get('content', '')
            if content:
                yield content

            if chunk.get('done'):
                # Yield final stats
                stats = {}
                for key in ('total_duration', 'eval_count', 'eval_duration',
                            'prompt_eval_count', 'prompt_eval_duration'):
                    if key in chunk:
                        stats[key] = chunk[key]
                if stats:
                    yield {'_stats': stats}
                return


def stream_generate(model: str, prompt: str, system: str = None,
                    options: dict = None, host: str = None):
    """Stream raw completion from Ollama. Yields content strings.

    Args:
        model: Ollama model name
        prompt: Raw prompt text
        system: Optional system prompt
        options: Ollama options dict
        host: Optional Ollama host URL

    Yields:
        str: Token chunks as they arrive
    """
    base = _get_ollama_base(host)
    body = {
        'model': model,
        'prompt': prompt,
        'stream': True,
        'options': options or {},
    }
    if system:
        body['system'] = system

    payload = json.dumps(body).encode()

    req = urllib.request.Request(
        f'{base}/api/generate',
        data=payload,
        headers={'Content-Type': 'application/json'},
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = chunk.get('response', '')
            if content:
                yield content

            if chunk.get('done'):
                stats = {}
                for key in ('total_duration', 'eval_count', 'eval_duration',
                            'prompt_eval_count', 'prompt_eval_duration'):
                    if key in chunk:
                        stats[key] = chunk[key]
                if stats:
                    yield {'_stats': stats}
                return


# ---------------------------------------------------------------------------
# Model registry — portable config for network model serving
# ---------------------------------------------------------------------------

REGISTRY_PATH = os.path.join(CONFIG_DIR, 'model_registry.yaml')


def load_model_registry() -> dict:
    """Load the model registry. Creates default if missing."""
    try:
        with open(REGISTRY_PATH) as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def save_model_registry(registry: dict):
    """Save the model registry."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def add_ollama_host(name: str, url: str, description: str = ''):
    """Register a remote Ollama host in the model registry."""
    registry = load_model_registry()
    if 'ollama_hosts' not in registry:
        registry['ollama_hosts'] = []

    # Update existing or add new
    for h in registry['ollama_hosts']:
        if h.get('name') == name:
            h['url'] = url
            h['description'] = description
            save_model_registry(registry)
            return
    registry['ollama_hosts'].append({
        'name': name,
        'url': url,
        'description': description,
    })
    save_model_registry(registry)


def remove_ollama_host(name: str) -> bool:
    """Remove a remote Ollama host from the registry."""
    registry = load_model_registry()
    hosts = registry.get('ollama_hosts', [])
    registry['ollama_hosts'] = [h for h in hosts if h.get('name') != name]
    save_model_registry(registry)
    return len(registry['ollama_hosts']) < len(hosts)
