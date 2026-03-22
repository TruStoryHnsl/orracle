"""Train/val split with optional stratification."""

from __future__ import annotations

import hashlib
import random

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType


@NodeRegistry.register
class SplitNode(BaseNode):
    node_type = 'train_val_split'
    label = 'Train/Val Split'
    category = 'encoding'
    description = 'Split data into training and validation sets with seeded shuffle'
    inputs = [Port('in', PortType.TEXT_BATCH, 'Text chunks')]
    outputs = [
        Port('train', PortType.TEXT_BATCH, 'Training set'),
        Port('val', PortType.TEXT_BATCH, 'Validation set'),
        Port('metrics', PortType.METRICS, 'Split stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'val_ratio': {
                'type': 'number',
                'title': 'Validation ratio',
                'default': 0.1,
                'minimum': 0.01,
                'maximum': 0.5,
            },
            'seed': {
                'type': 'integer',
                'title': 'Random seed',
                'default': 42,
            },
            'stratify_by': {
                'type': 'string',
                'title': 'Stratify by',
                'description': 'Metadata key for stratified split (empty = random)',
                'default': '',
            },
            'deterministic': {
                'type': 'boolean',
                'title': 'Deterministic (hash-based)',
                'description': 'Use content hash instead of random shuffle for reproducibility',
                'default': False,
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('in', [])
        val_ratio = config.get('val_ratio', 0.1)
        seed = config.get('seed', 42)
        stratify_key = config.get('stratify_by', '').strip()
        deterministic = config.get('deterministic', False)

        if deterministic:
            # Hash-based split: consistent regardless of order
            train, val = [], []
            for chunk in chunks:
                h = hashlib.md5(chunk.text[:1000].encode()).hexdigest()
                # Use first 8 hex chars as fraction
                frac = int(h[:8], 16) / 0xFFFFFFFF
                if frac < val_ratio:
                    val.append(chunk)
                else:
                    train.append(chunk)
        elif stratify_key:
            # Stratified split: maintain category proportions
            groups = {}
            for chunk in chunks:
                key = chunk.metadata.get(stratify_key, '_unknown')
                groups.setdefault(key, []).append(chunk)

            train, val = [], []
            rng = random.Random(seed)
            for key, group in groups.items():
                rng.shuffle(group)
                n_val = max(1, int(len(group) * val_ratio))
                val.extend(group[:n_val])
                train.extend(group[n_val:])
        else:
            # Simple random split
            shuffled = list(chunks)
            random.Random(seed).shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_ratio))
            val = shuffled[:n_val]
            train = shuffled[n_val:]

        train_chars = sum(len(c.text) for c in train)
        val_chars = sum(len(c.text) for c in val)

        return {
            'train': train,
            'val': val,
            'metrics': {
                'total': len(chunks),
                'train_count': len(train),
                'val_count': len(val),
                'actual_val_ratio': round(len(val) / max(1, len(chunks)), 3),
                'train_chars': train_chars,
                'val_chars': val_chars,
            },
        }
