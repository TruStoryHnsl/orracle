"""Key-moment detector trainer — multi-modal late fusion over clips.

Consumes the output of `run_video_pipeline.py` (frames, ROI crops, pose keypoints,
audio spectrograms, motion profiles) and trains a classifier to detect key moments
in cbsr stream recordings.

Designed for marathon-length runs (55+ hours) with granular checkpointing:
state is persisted every N steps so the process can be stopped at any time
(SIGINT, reboot, power loss) and resumed from the latest checkpoint on relaunch.

Modules:
    dataset   — MultiModalDataset + collate_fn
    model     — KeyMomentDetector and per-modality encoders
    train     — Resumable training loop
"""

from training.video import dataset, model, train

__all__ = ['dataset', 'model', 'train']
