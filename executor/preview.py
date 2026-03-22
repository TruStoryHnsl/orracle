"""PreviewManager — caching and diff generation for node previews."""

from __future__ import annotations

import difflib
import hashlib
import time
from typing import Any


class PreviewCache:
    """Simple TTL cache for node preview results."""

    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self._cache: dict[str, dict] = {}

    def _key(self, pipeline_id: str, node_id: str, config: dict) -> str:
        config_str = str(sorted(config.items())) if config else ''
        return hashlib.md5(
            f'{pipeline_id}:{node_id}:{config_str}'.encode()
        ).hexdigest()

    def get(self, pipeline_id: str, node_id: str, config: dict) -> dict | None:
        key = self._key(pipeline_id, node_id, config)
        entry = self._cache.get(key)
        if entry and time.time() - entry['time'] < self.ttl:
            return entry['data']
        return None

    def set(self, pipeline_id: str, node_id: str, config: dict, data: dict):
        key = self._key(pipeline_id, node_id, config)
        self._cache[key] = {'data': data, 'time': time.time()}

    def invalidate(self, pipeline_id: str, node_id: str | None = None):
        """Invalidate cache for a node or entire pipeline."""
        prefix = f'{pipeline_id}:{node_id}:' if node_id else f'{pipeline_id}:'
        # Need to reconstruct keys, so just clear all matching entries
        to_remove = []
        for key, entry in self._cache.items():
            # Simple approach: clear all if we can't match by prefix
            # (keys are hashed, so we clear aggressively)
            to_remove.append(key)
        if node_id:
            # Just clear all — cache is cheap to rebuild
            self._cache.clear()
        else:
            self._cache.clear()

    def clear(self):
        self._cache.clear()


def generate_diff(before: str, after: str, context_lines: int = 3) -> list[dict]:
    """Generate a character-level diff between before and after text.
    Returns list of {type: 'equal'|'insert'|'delete', text: str} segments.
    """
    # Use unified diff for line-level view
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile='before', tofile='after',
        n=context_lines,
    ))

    segments = []
    for line in diff[2:]:  # Skip --- and +++ headers
        if line.startswith('+'):
            segments.append({'type': 'insert', 'text': line[1:]})
        elif line.startswith('-'):
            segments.append({'type': 'delete', 'text': line[1:]})
        elif line.startswith(' '):
            segments.append({'type': 'equal', 'text': line[1:]})
        elif line.startswith('@@'):
            segments.append({'type': 'header', 'text': line.strip()})

    return segments


def compute_stats(chunks: list) -> dict:
    """Compute aggregate stats for a list of DataChunks."""
    if not chunks:
        return {'count': 0}

    texts = [c.text if hasattr(c, 'text') else str(c) for c in chunks]
    total_chars = sum(len(t) for t in texts)
    total_words = sum(len(t.split()) for t in texts)
    total_lines = sum(t.count('\n') + 1 for t in texts)

    return {
        'count': len(chunks),
        'total_chars': total_chars,
        'total_words': total_words,
        'total_lines': total_lines,
        'avg_chars': round(total_chars / len(chunks)),
        'avg_words': round(total_words / len(chunks)),
        'min_chars': min(len(t) for t in texts),
        'max_chars': max(len(t) for t in texts),
    }
