#!/usr/bin/env python3
"""Run the key-moment detector trainer.

Multi-modal late-fusion classifier trained on the output of
`run_video_pipeline.py`. Designed for marathon runs (55+ hours) with
granular checkpointing: stop anytime with Ctrl-C, relaunch with --resume.

Usage:
    # Fresh run with the default config
    python run_video_train.py

    # Custom config
    python run_video_train.py --config training/video/config.yaml

    # Resume from last checkpoint
    python run_video_train.py --resume

    # Short smoke test
    python run_video_train.py --max-steps 100 --device cpu

Checkpoints are written to <ckpt_dir>/ with last.pt (most recent) and
best.pt (lowest validation loss). Older rolling checkpoints are rotated
out to keep disk usage bounded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
VENV_PYTHON = SCRIPT_DIR / 'venv' / 'bin' / 'python'
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

# Default to the bundled config unless the user passes --config explicitly.
_DEFAULT_CONFIG = SCRIPT_DIR / 'training' / 'video' / 'config.yaml'
if '--config' not in sys.argv and _DEFAULT_CONFIG.exists():
    sys.argv.extend(['--config', str(_DEFAULT_CONFIG)])

from training.video.train import main  # noqa: E402

if __name__ == '__main__':
    main()
