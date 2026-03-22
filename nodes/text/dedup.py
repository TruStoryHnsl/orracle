"""Exact (hash) and fuzzy deduplication of text chunks."""

from __future__ import annotations

import hashlib
import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType


def _text_hash(text: str) -> str:
    """SHA-256 hash of normalized text."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _ngram_fingerprint(text: str, n: int = 5) -> set[str]:
    """Generate character n-gram set for fuzzy comparison."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    if len(normalized) < n:
        return {normalized}
    return {normalized[i:i+n] for i in range(len(normalized) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@NodeRegistry.register
class DedupNode(BaseNode):
    node_type = 'dedup'
    label = 'Deduplicate'
    category = 'text'
    description = 'Remove duplicate text chunks (exact hash or fuzzy similarity)'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text chunks')]
    outputs = [
        Port('unique', PortType.TEXT_BATCH, 'Deduplicated chunks'),
        Port('dupes', PortType.TEXT_BATCH, 'Removed duplicates', required=False),
        Port('metrics', PortType.METRICS, 'Dedup stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'mode': {
                'type': 'string',
                'title': 'Mode',
                'enum': ['exact', 'fuzzy', 'both'],
                'default': 'exact',
            },
            'fuzzy_threshold': {
                'type': 'number',
                'title': 'Fuzzy threshold',
                'description': 'Jaccard similarity threshold (0-1) for fuzzy dedup',
                'default': 0.85,
                'minimum': 0.5,
                'maximum': 1.0,
            },
            'ngram_size': {
                'type': 'integer',
                'title': 'N-gram size',
                'description': 'Character n-gram size for fingerprinting',
                'default': 5,
                'minimum': 3,
                'maximum': 10,
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        mode = config.get('mode', 'exact')
        threshold = config.get('fuzzy_threshold', 0.85)
        ngram_size = config.get('ngram_size', 5)

        unique = []
        dupes = []
        seen_hashes = set()
        fingerprints = []

        for chunk in chunks:
            is_dupe = False

            # Exact dedup
            if mode in ('exact', 'both'):
                h = _text_hash(chunk.text)
                if h in seen_hashes:
                    is_dupe = True
                seen_hashes.add(h)

            # Fuzzy dedup
            if not is_dupe and mode in ('fuzzy', 'both'):
                fp = _ngram_fingerprint(chunk.text, ngram_size)
                for existing_fp in fingerprints:
                    if _jaccard(fp, existing_fp) >= threshold:
                        is_dupe = True
                        break
                if not is_dupe:
                    fingerprints.append(fp)

            if is_dupe:
                dupes.append(chunk)
            else:
                unique.append(chunk)

        return {
            'unique': unique,
            'out': unique,
            'dupes': dupes,
            'metrics': {
                'total': len(chunks),
                'unique': len(unique),
                'duplicates': len(dupes),
                'dedup_rate': round(len(dupes) / max(1, len(chunks)) * 100, 1),
            },
        }
