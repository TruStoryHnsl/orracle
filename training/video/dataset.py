"""Multi-modal dataset for the key-moment detector.

Consumes the output layout of `run_video_pipeline.py`:

    <root>/
    ├── train.jsonl / val.jsonl   — per-frame records
    ├── frames/<basename>.jpg
    ├── roi/<basename>_roi.jpg
    ├── pose/<basename>_pose.json
    ├── audio/<set_key>-<ct>_{km|negN}_spec.png
    └── motion/<set_key>-<ct>_motion.json

Each JSONL record has::

    {
      "label": "key_moment" | "not_key_moment" | "near_key_moment",
      "image_path": ".../frames/<basename>.jpg",
      "timestamp": float,
      "streamer": str,
      "clip_type": int,
      "key_moment_offset": float
    }

The dataset resolves sibling modality files from the frame's basename. If a
modality is missing (e.g. a negative frame with no per-frame audio spectrogram),
it emits a zero tensor plus a boolean `*_present` flag so the model can route
through a learned "missing" embedding.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_TO_IDX = {
    'key_moment': 0,
    'near_key_moment': 1,
    'not_key_moment': 2,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABEL_TO_IDX)

# 17 COCO keypoints × (x, y, visibility)
POSE_FEATURES = 17 * 3

# Motion curve target length (from motion_analyzer: 10s window × 2fps = 20 steps).
MOTION_STEPS = 20

# Spectrogram target size (H×W). The raw spectrograms are variable size; we
# resize so every sample is the same shape.
SPEC_SIZE = (128, 256)

# ROI crop target size.
ROI_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------

# Frame filename convention from frame_extractor.py:
#   {set_key}-{clip_type}_km.jpg         → positive
#   {set_key}-{clip_type}_neg{N}.jpg     → negative
#   {set_key}-{clip_type}_seq{±N}.jpg    → near
_NAME_RE = re.compile(r'^(.+)-(\d)_(km|neg\d+|seq[+-]?\d+)\.jpg$')


@dataclass
class FrameKey:
    """Parsed identifiers for locating sibling modality files."""
    basename: str       # filename without extension
    set_key: str        # YYMMDD-HHMMSS-streamer
    clip_type: int      # 1/2/3
    kind: str           # 'km' | 'negN' | 'seqN'

    @property
    def clip_id(self) -> str:
        """Per-clip key used to locate audio and motion files."""
        return f'{self.set_key}-{self.clip_type}'

    @property
    def audio_tag(self) -> str:
        """Tag inside the audio filename: 'km' or 'neg0'/'neg1' for negatives.

        Sequence frames don't have their own audio — we fall back to 'km'
        since sequence frames are temporally close to the key moment.
        """
        if self.kind.startswith('seq'):
            return 'km'
        return self.kind  # 'km' or 'negN'


def _parse_frame_name(image_path: str) -> FrameKey | None:
    name = os.path.basename(image_path)
    m = _NAME_RE.match(name)
    if not m:
        return None
    set_key = m.group(1)
    clip_type = int(m.group(2))
    kind = m.group(3)
    basename = name.rsplit('.', 1)[0]
    return FrameKey(basename=basename, set_key=set_key,
                    clip_type=clip_type, kind=kind)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    """PyTorch Dataset producing aligned (ROI, spec, pose, motion) tensors.

    Args:
        jsonl_path: Path to train.jsonl or val.jsonl.
        root: Root directory holding the sibling modality dirs
              (frames/, roi/, pose/, audio/, motion/).
        train: If True, enables light augmentation on ROI crops.
    """

    def __init__(self, jsonl_path: str, root: str | None = None,
                 train: bool = True):
        self.jsonl_path = jsonl_path
        self.root = root or os.path.dirname(os.path.abspath(jsonl_path))
        self.train_mode = train

        self.records: list[dict[str, Any]] = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'image_path' not in rec or 'label' not in rec:
                    continue
                if rec['label'] not in LABEL_TO_IDX:
                    continue
                self.records.append(rec)

        if not self.records:
            raise ValueError(f'No usable records in {jsonl_path}')

        # Subdir layout
        self.roi_dir = os.path.join(self.root, 'roi')
        self.pose_dir = os.path.join(self.root, 'pose')
        self.audio_dir = os.path.join(self.root, 'audio')
        self.motion_dir = os.path.join(self.root, 'motion')

    # ------------------------------------------------------------------
    # Length & label stats
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r['label']] = counts.get(r['label'], 0) + 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss."""
        counts = self.label_counts()
        total = sum(counts.values())
        weights = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for label, idx in LABEL_TO_IDX.items():
            n = counts.get(label, 0)
            weights[idx] = total / (NUM_CLASSES * n) if n > 0 else 0.0
        return weights

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        label_idx = LABEL_TO_IDX[rec['label']]
        key = _parse_frame_name(rec['image_path'])

        roi, roi_ok = self._load_roi(rec, key)
        spec, spec_ok = self._load_spectrogram(key)
        pose, pose_ok = self._load_pose(rec, key)
        motion, motion_ok = self._load_motion(key)

        return {
            'roi': roi,
            'roi_present': torch.tensor(roi_ok, dtype=torch.bool),
            'spec': spec,
            'spec_present': torch.tensor(spec_ok, dtype=torch.bool),
            'pose': pose,
            'pose_present': torch.tensor(pose_ok, dtype=torch.bool),
            'motion': motion,
            'motion_present': torch.tensor(motion_ok, dtype=torch.bool),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'streamer': rec.get('streamer', 'unknown'),
        }

    # ------------------------------------------------------------------
    # Per-modality loaders (return (tensor, present_flag))
    # ------------------------------------------------------------------
    def _load_roi(self, rec: dict, key: FrameKey | None) -> tuple[torch.Tensor, bool]:
        roi_path = None
        if key is not None:
            roi_path = os.path.join(self.roi_dir, f'{key.basename}_roi.jpg')
            if not os.path.isfile(roi_path):
                roi_path = None
        # Fall back to the full frame if the ROI crop is missing.
        if roi_path is None:
            frame_path = rec['image_path']
            if not os.path.isabs(frame_path):
                frame_path = os.path.join(self.root, frame_path)
            if os.path.isfile(frame_path):
                roi_path = frame_path

        if roi_path is None:
            return torch.zeros(3, *ROI_SIZE, dtype=torch.float32), False

        try:
            img = Image.open(roi_path).convert('RGB').resize(
                ROI_SIZE, Image.BILINEAR)
        except (OSError, ValueError):
            return torch.zeros(3, *ROI_SIZE, dtype=torch.float32), False

        arr = np.asarray(img, dtype=np.float32) / 255.0  # H×W×3
        if self.train_mode and np.random.rand() < 0.5:
            arr = arr[:, ::-1, :].copy()  # horizontal flip

        # Normalize to [-1, 1] — avoids depending on ImageNet stats since many
        # ROI crops are much tighter and brighter than natural images.
        arr = (arr - 0.5) * 2.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor, True

    def _load_spectrogram(self, key: FrameKey | None) -> tuple[torch.Tensor, bool]:
        if key is None:
            return torch.zeros(1, *SPEC_SIZE, dtype=torch.float32), False
        spec_path = os.path.join(
            self.audio_dir, f'{key.clip_id}_{key.audio_tag}_spec.png')
        if not os.path.isfile(spec_path):
            return torch.zeros(1, *SPEC_SIZE, dtype=torch.float32), False
        try:
            img = Image.open(spec_path).convert('L').resize(
                (SPEC_SIZE[1], SPEC_SIZE[0]), Image.BILINEAR)
        except (OSError, ValueError):
            return torch.zeros(1, *SPEC_SIZE, dtype=torch.float32), False
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) * 2.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # 1×H×W
        return tensor, True

    def _load_pose(self, rec: dict,
                   key: FrameKey | None) -> tuple[torch.Tensor, bool]:
        if key is None:
            return torch.zeros(POSE_FEATURES, dtype=torch.float32), False
        pose_path = os.path.join(self.pose_dir, f'{key.basename}_pose.json')
        if not os.path.isfile(pose_path):
            return torch.zeros(POSE_FEATURES, dtype=torch.float32), False
        try:
            with open(pose_path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return torch.zeros(POSE_FEATURES, dtype=torch.float32), False

        landmarks = data.get('landmarks') or []
        if not landmarks:
            return torch.zeros(POSE_FEATURES, dtype=torch.float32), False

        flat = np.zeros(POSE_FEATURES, dtype=np.float32)
        for i, lm in enumerate(landmarks[:17]):
            flat[i * 3 + 0] = float(lm.get('x', 0.0))
            flat[i * 3 + 1] = float(lm.get('y', 0.0))
            flat[i * 3 + 2] = float(lm.get('visibility', 0.0))
        return torch.from_numpy(flat), True

    def _load_motion(self, key: FrameKey | None) -> tuple[torch.Tensor, bool]:
        if key is None:
            return torch.zeros(1, MOTION_STEPS, dtype=torch.float32), False
        motion_path = os.path.join(
            self.motion_dir, f'{key.clip_id}_motion.json')
        if not os.path.isfile(motion_path):
            return torch.zeros(1, MOTION_STEPS, dtype=torch.float32), False
        try:
            with open(motion_path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return torch.zeros(1, MOTION_STEPS, dtype=torch.float32), False

        curve = data.get('motion_curve') or []
        if not curve:
            return torch.zeros(1, MOTION_STEPS, dtype=torch.float32), False

        diffs = np.array([float(p.get('diff', 0.0)) for p in curve],
                         dtype=np.float32)
        # Pad or crop to MOTION_STEPS (center the curve on key moment).
        if len(diffs) >= MOTION_STEPS:
            start = (len(diffs) - MOTION_STEPS) // 2
            diffs = diffs[start:start + MOTION_STEPS]
        else:
            pad = MOTION_STEPS - len(diffs)
            left = pad // 2
            right = pad - left
            diffs = np.pad(diffs, (left, right), mode='constant')
        # Normalize by the clip's own peak so scale is consistent across clips.
        peak = max(float(data.get('peak_motion_value', diffs.max())), 1e-6)
        diffs = diffs / peak
        return torch.from_numpy(diffs).unsqueeze(0), True  # 1×MOTION_STEPS


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack samples into a batch dict.

    Keeps `streamer` as a plain list so we don't lose the identifier, and
    stacks every tensor field with torch.stack.
    """
    out: dict[str, Any] = {}
    tensor_keys = ['roi', 'roi_present', 'spec', 'spec_present',
                   'pose', 'pose_present', 'motion', 'motion_present', 'label']
    for k in tensor_keys:
        out[k] = torch.stack([s[k] for s in batch])
    out['streamer'] = [s['streamer'] for s in batch]
    return out
