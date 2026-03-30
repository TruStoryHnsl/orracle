"""Shared utilities for orracle blueprints.

Consolidates helpers that were duplicated across multiple blueprints:
machine config loading, config dir resolution, and common patterns.
"""

from __future__ import annotations

import os
import threading
import time

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')

# ---------------------------------------------------------------------------
# Machine config
# ---------------------------------------------------------------------------

def load_machines() -> dict:
    """Load all machines from machines.yaml."""
    try:
        with open(os.path.join(CONFIG_DIR, 'machines.yaml')) as f:
            data = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        data = {}
    return data.get('machines', {})


def save_machines(machines: dict):
    """Persist machines dict to machines.yaml."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(os.path.join(CONFIG_DIR, 'machines.yaml'), 'w') as f:
        yaml.dump({'machines': machines}, f, default_flow_style=False,
                  sort_keys=False)


def get_machine(name: str) -> dict | None:
    """Get a single machine config by name, or None."""
    return load_machines().get(name)


# ---------------------------------------------------------------------------
# Hardware cache (shared across blueprints)
# ---------------------------------------------------------------------------

_hw_cache: dict = {'data': None, 'ts': 0}
_hw_lock = threading.Lock()
HW_CACHE_TTL = 300  # 5 minutes


def get_local_hardware() -> dict:
    """Return cached local hardware info, refreshing after TTL."""
    from training import hardware
    with _hw_lock:
        if _hw_cache['data'] and time.time() - _hw_cache['ts'] < HW_CACHE_TTL:
            return _hw_cache['data']
        hw = hardware.detect_hardware()
        _hw_cache['data'] = hw
        _hw_cache['ts'] = time.time()
        return hw


def refresh_hardware() -> dict:
    """Force-refresh local hardware cache."""
    with _hw_lock:
        _hw_cache['data'] = None
        _hw_cache['ts'] = 0
    return get_local_hardware()


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------

def load_yaml(filename: str) -> dict:
    """Load a YAML config file from CONFIG_DIR. Returns {} on error."""
    try:
        with open(os.path.join(CONFIG_DIR, filename)) as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def save_yaml(filename: str, data: dict):
    """Save data to a YAML config file in CONFIG_DIR."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(os.path.join(CONFIG_DIR, filename), 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
