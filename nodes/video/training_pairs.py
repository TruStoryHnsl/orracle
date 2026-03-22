"""Training Pair Generator — packages extracted frames into training-ready formats.

Supports multiple output formats:
  - JSONL with base64-encoded images (portable, self-contained)
  - Directory + labels CSV (for frameworks that load images from disk)
  - CLIP-style pairs (image path + text description)

Also generates dataset statistics and train/val splits.
"""

from __future__ import annotations

import base64
import json
import os
import random

from nodes.base import BaseNode, NodeRegistry, Port, PortType
from nodes.video.frame_extractor import ExtractedFrame


def _image_to_base64(path: str) -> str | None:
    """Read an image file and return base64-encoded string."""
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('ascii')
    except OSError:
        return None


def export_jsonl(frames: list[ExtractedFrame], output_path: str,
                 include_base64: bool = True) -> dict:
    """Export frames as JSONL with labels and optional base64 images.

    Each line: {"image": "base64...", "label": "key_moment", "metadata": {...}}
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for frame in frames:
            record = {
                'label': frame.label,
                'timestamp': frame.timestamp,
                'streamer': frame.streamer,
                'clip_type': frame.clip_type,
                'key_moment_offset': frame.key_moment_offset,
                'image_path': frame.image_path,
            }
            if include_base64:
                b64 = _image_to_base64(frame.image_path)
                if b64 is None:
                    continue
                record['image_base64'] = b64

            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

    return {'path': output_path, 'count': count}


def export_csv(frames: list[ExtractedFrame], output_path: str) -> dict:
    """Export frames as a CSV labels file (image_path, label, metadata)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('image_path,label,timestamp,streamer,clip_type,key_moment_offset\n')
        for frame in frames:
            f.write(f'{frame.image_path},{frame.label},{frame.timestamp},'
                    f'{frame.streamer},{frame.clip_type},{frame.key_moment_offset}\n')

    return {'path': output_path, 'count': len(frames)}


def train_val_split(frames: list[ExtractedFrame], val_ratio: float = 0.15,
                    seed: int = 42, stratify_by_streamer: bool = True
                    ) -> tuple[list[ExtractedFrame], list[ExtractedFrame]]:
    """Split frames into train and validation sets.

    If stratify_by_streamer, ensures each streamer's clips are entirely
    in train or val (no data leakage between splits).
    """
    rng = random.Random(seed)

    if stratify_by_streamer:
        # Group by streamer, split streamers into train/val
        streamers: dict[str, list[ExtractedFrame]] = {}
        for f in frames:
            streamers.setdefault(f.streamer, []).append(f)

        streamer_list = sorted(streamers.keys())
        rng.shuffle(streamer_list)

        n_val_streamers = max(1, int(len(streamer_list) * val_ratio))
        val_streamers = set(streamer_list[:n_val_streamers])

        train = [f for f in frames if f.streamer not in val_streamers]
        val = [f for f in frames if f.streamer in val_streamers]
    else:
        shuffled = list(frames)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val = shuffled[:n_val]
        train = shuffled[n_val:]

    return train, val


@NodeRegistry.register
class TrainingPairNode(BaseNode):
    node_type = 'video_training_pairs'
    label = 'Training Pairs'
    category = 'video'
    description = 'Export extracted frames as training data (JSONL or CSV with train/val split)'
    inputs = [
        Port('frames', PortType.FRAMES, 'Extracted frames with labels'),
    ]
    outputs = [
        Port('out', PortType.METRICS, 'Export results and statistics'),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'output_dir': {
                'type': 'string',
                'title': 'Output Directory',
                'default': './output/video_training',
            },
            'format': {
                'type': 'string',
                'title': 'Export Format',
                'enum': ['jsonl', 'csv', 'both'],
                'default': 'both',
            },
            'include_base64': {
                'type': 'boolean',
                'title': 'Include Base64 Images in JSONL',
                'default': False,
            },
            'val_ratio': {
                'type': 'number',
                'title': 'Validation Ratio',
                'default': 0.15, 'minimum': 0.0, 'maximum': 0.5,
            },
            'stratify_by_streamer': {
                'type': 'boolean',
                'title': 'Stratify by Streamer',
                'description': 'Keep all clips from a streamer in same split (prevents leakage)',
                'default': True,
            },
        },
    }

    def process(self, inputs, config):
        frames = inputs.get('frames', [])
        output_dir = config.get('output_dir', './output/video_training')
        fmt = config.get('format', 'both')
        include_b64 = config.get('include_base64', False)
        val_ratio = config.get('val_ratio', 0.15)
        stratify = config.get('stratify_by_streamer', True)

        os.makedirs(output_dir, exist_ok=True)

        # Split
        train_frames, val_frames = train_val_split(
            frames, val_ratio, stratify_by_streamer=stratify)

        results = {}

        if fmt in ('jsonl', 'both'):
            train_jsonl = export_jsonl(
                train_frames,
                os.path.join(output_dir, 'train.jsonl'),
                include_base64=include_b64)
            val_jsonl = export_jsonl(
                val_frames,
                os.path.join(output_dir, 'val.jsonl'),
                include_base64=include_b64)
            results['jsonl'] = {
                'train': train_jsonl,
                'val': val_jsonl,
            }

        if fmt in ('csv', 'both'):
            train_csv = export_csv(
                train_frames,
                os.path.join(output_dir, 'train.csv'))
            val_csv = export_csv(
                val_frames,
                os.path.join(output_dir, 'val.csv'))
            results['csv'] = {
                'train': train_csv,
                'val': val_csv,
            }

        # Statistics
        pos_train = sum(1 for f in train_frames if f.label == 'key_moment')
        neg_train = sum(1 for f in train_frames if f.label == 'not_key_moment')
        pos_val = sum(1 for f in val_frames if f.label == 'key_moment')
        neg_val = sum(1 for f in val_frames if f.label == 'not_key_moment')

        return {
            'out': {
                'total_frames': len(frames),
                'train_frames': len(train_frames),
                'val_frames': len(val_frames),
                'train_positive': pos_train,
                'train_negative': neg_train,
                'val_positive': pos_val,
                'val_negative': neg_val,
                'train_streamers': len(set(f.streamer for f in train_frames)),
                'val_streamers': len(set(f.streamer for f in val_frames)),
                'exports': results,
            },
        }
