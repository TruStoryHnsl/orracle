"""Video processing nodes — multi-modal key moment detection pipeline.

Extracts four parallel data streams from labeled clips:
  - Visual frames (image classifier)
  - Audio segments + spectrograms (audio classifier)
  - Body pose keypoints (pose classifier)
  - Motion profiles (temporal change detector)
"""

from nodes.video import clip_scanner
from nodes.video import frame_extractor
from nodes.video import audio_extractor
from nodes.video import pose_extractor
from nodes.video import roi_extractor
from nodes.video import motion_analyzer
from nodes.video import training_pairs
