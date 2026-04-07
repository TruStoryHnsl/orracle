"""Key-moment detector — multi-modal late fusion.

Four independent encoders process each modality into a fixed-dim embedding,
the embeddings are concatenated, and a small MLP head emits 3-class logits
(key_moment / near_key_moment / not_key_moment).

Each modality carries a boolean `present` flag. When a modality is absent,
its encoder output is replaced with a learned "missing" embedding of the
same dimension, so the fusion head sees a consistent shape regardless of
which streams a given sample has.
"""

from __future__ import annotations

import torch
from torch import nn

from training.video.dataset import (MOTION_STEPS, NUM_CLASSES, POSE_FEATURES,
                                    ROI_SIZE, SPEC_SIZE)

# ---------------------------------------------------------------------------
# Embedding dimensions per modality
# ---------------------------------------------------------------------------
ROI_DIM = 128
SPEC_DIM = 64
POSE_DIM = 64
MOTION_DIM = 32
FUSION_DIM = ROI_DIM + SPEC_DIM + POSE_DIM + MOTION_DIM  # 288


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → optional pool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


# ---------------------------------------------------------------------------
# Per-modality encoders
# ---------------------------------------------------------------------------

class ROIEncoder(nn.Module):
    """Small CNN over the wrist-hip ROI crop (3×224×224 → ROI_DIM)."""

    def __init__(self, out_dim: int = ROI_DIM):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32),    # 112
            ConvBlock(32, 64),   # 56
            ConvBlock(64, 128),  # 28
            ConvBlock(128, 256), # 14
            ConvBlock(256, 256), # 7
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class SpecEncoder(nn.Module):
    """Small CNN over the mel spectrogram (1×H×W → SPEC_DIM)."""

    def __init__(self, out_dim: int = SPEC_DIM):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class PoseEncoder(nn.Module):
    """MLP over flattened 17×3 keypoints (POSE_FEATURES → POSE_DIM)."""

    def __init__(self, out_dim: int = POSE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(POSE_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MotionEncoder(nn.Module):
    """1D conv over the motion curve (1×MOTION_STEPS → MOTION_DIM)."""

    def __init__(self, out_dim: int = MOTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class KeyMomentDetector(nn.Module):
    """Multi-modal late-fusion classifier.

    Missing modalities are replaced by learned "absent" embeddings (a single
    parameter vector per modality) so the fusion MLP always sees a consistent
    shape, and the model can learn that "audio missing" is itself a signal.
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.2):
        super().__init__()
        self.roi_enc = ROIEncoder()
        self.spec_enc = SpecEncoder()
        self.pose_enc = PoseEncoder()
        self.motion_enc = MotionEncoder()

        # Learned "missing" embeddings — one per modality.
        self.missing_roi = nn.Parameter(torch.zeros(ROI_DIM))
        self.missing_spec = nn.Parameter(torch.zeros(SPEC_DIM))
        self.missing_pose = nn.Parameter(torch.zeros(POSE_DIM))
        self.missing_motion = nn.Parameter(torch.zeros(MOTION_DIM))

        self.fusion = nn.Sequential(
            nn.Linear(FUSION_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _gate(enc: torch.Tensor, present: torch.Tensor,
              missing: nn.Parameter) -> torch.Tensor:
        """Replace rows where present=False with the learned missing vector."""
        mask = present.float().unsqueeze(-1)   # (B, 1)
        miss = missing.unsqueeze(0).expand_as(enc)  # (B, D)
        return enc * mask + miss * (1.0 - mask)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        roi = self._gate(self.roi_enc(batch['roi']),
                         batch['roi_present'], self.missing_roi)
        spec = self._gate(self.spec_enc(batch['spec']),
                          batch['spec_present'], self.missing_spec)
        pose = self._gate(self.pose_enc(batch['pose']),
                          batch['pose_present'], self.missing_pose)
        motion = self._gate(self.motion_enc(batch['motion']),
                            batch['motion_present'], self.missing_motion)
        fused = torch.cat([roi, spec, pose, motion], dim=-1)
        return self.fusion(fused)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity helper (used by tests & startup log)
# ---------------------------------------------------------------------------

def dummy_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Produce a synthetic batch matching the dataset contract."""
    return {
        'roi': torch.zeros(batch_size, 3, *ROI_SIZE),
        'roi_present': torch.ones(batch_size, dtype=torch.bool),
        'spec': torch.zeros(batch_size, 1, *SPEC_SIZE),
        'spec_present': torch.ones(batch_size, dtype=torch.bool),
        'pose': torch.zeros(batch_size, POSE_FEATURES),
        'pose_present': torch.ones(batch_size, dtype=torch.bool),
        'motion': torch.zeros(batch_size, 1, MOTION_STEPS),
        'motion_present': torch.ones(batch_size, dtype=torch.bool),
    }
