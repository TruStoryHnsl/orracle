"""Frame Extractor — extracts frames from clips at specific timestamps.

Given clips with key_moment metadata, extracts:
  - Positive frames at the key_moment timestamp
  - Negative frames at random offsets (non-key-moment timestamps)
  - Optional: sequence of frames around the key moment for temporal context

Uses ffmpeg for frame extraction — fast, no Python video decoding needed.
"""

from __future__ import annotations

import os
import random
import subprocess
from dataclasses import dataclass, field

from nodes.base import BaseNode, NodeRegistry, Port, PortType
from nodes.video.clip_scanner import ClipInfo


@dataclass
class ExtractedFrame:
    """A single extracted frame with its label and metadata."""
    image_path: str              # Path to the saved frame (JPEG)
    timestamp: float             # Seconds into the clip
    label: str                   # 'key_moment' or 'not_key_moment'
    clip_path: str               # Source clip
    streamer: str
    set_key: str
    clip_type: int
    key_moment_offset: float     # Distance from key_moment in seconds (0 = at key moment)

    def to_dict(self) -> dict:
        return {
            'image_path': self.image_path,
            'timestamp': self.timestamp,
            'label': self.label,
            'clip_path': self.clip_path,
            'streamer': self.streamer,
            'set_key': self.set_key,
            'clip_type': self.clip_type,
            'key_moment_offset': self.key_moment_offset,
        }


def _extract_frame(clip_path: str, timestamp: float, output_path: str,
                   quality: int = 2, timeout: int = 15) -> bool:
    """Extract a single frame from a video at the given timestamp.

    Args:
        clip_path: Path to MP4 file
        timestamp: Seconds into the video
        output_path: Where to save the JPEG
        quality: JPEG quality (2 = high, 31 = low)
        timeout: ffmpeg timeout in seconds

    Returns:
        True if extraction succeeded
    """
    try:
        r = subprocess.run(
            ['ffmpeg', '-y', '-ss', str(timestamp),
             '-i', clip_path,
             '-frames:v', '1',
             '-q:v', str(quality),
             output_path],
            capture_output=True, timeout=timeout,
        )
        return r.returncode == 0 and os.path.exists(output_path)
    except subprocess.TimeoutExpired:
        return False


def extract_training_frames(
    clips: list[ClipInfo],
    output_dir: str,
    negatives_per_clip: int = 3,
    sequence_frames: int = 0,
    sequence_interval: float = 1.0,
    seed: int = 42,
) -> list[ExtractedFrame]:
    """Extract positive and negative frames from clips.

    For each clip with a key_moment:
      - Extract 1 frame at the key_moment (positive)
      - Extract N frames at random timestamps (negatives)
      - Optionally extract a sequence around the key moment

    Args:
        clips: List of ClipInfo with key_moment metadata
        output_dir: Directory to save extracted frames
        negatives_per_clip: How many negative frames per clip
        sequence_frames: How many frames before/after key_moment (0 = disabled)
        sequence_interval: Seconds between sequence frames
        seed: Random seed for reproducible negative sampling

    Returns:
        List of ExtractedFrame objects
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    frames = []

    labeled_clips = [c for c in clips if c.key_moment is not None and c.duration > 0]

    for clip in labeled_clips:
        km = clip.key_moment
        dur = clip.duration
        base = f'{clip.set_key}-{clip.clip_type}'

        # Positive: frame at key moment
        pos_path = os.path.join(output_dir, f'{base}_km.jpg')
        if _extract_frame(clip.path, km, pos_path):
            frames.append(ExtractedFrame(
                image_path=pos_path,
                timestamp=km,
                label='key_moment',
                clip_path=clip.path,
                streamer=clip.streamer,
                set_key=clip.set_key,
                clip_type=clip.clip_type,
                key_moment_offset=0.0,
            ))

        # Negatives: random timestamps at least 5s away from key moment
        margin = min(5.0, dur * 0.1)  # At least 5s or 10% of duration
        neg_candidates = []
        if km - margin > 1:
            neg_candidates.append((1.0, km - margin))
        if km + margin < dur - 1:
            neg_candidates.append((km + margin, dur - 1))

        for neg_idx in range(negatives_per_clip):
            if not neg_candidates:
                break
            # Pick a random range, then a random time within it
            low, high = rng.choice(neg_candidates)
            if high <= low:
                continue
            ts = rng.uniform(low, high)
            neg_path = os.path.join(output_dir, f'{base}_neg{neg_idx}.jpg')
            if _extract_frame(clip.path, ts, neg_path):
                frames.append(ExtractedFrame(
                    image_path=neg_path,
                    timestamp=ts,
                    label='not_key_moment',
                    clip_path=clip.path,
                    streamer=clip.streamer,
                    set_key=clip.set_key,
                    clip_type=clip.clip_type,
                    key_moment_offset=ts - km,
                ))

        # Sequence around key moment (for temporal models)
        if sequence_frames > 0:
            for i in range(-sequence_frames, sequence_frames + 1):
                ts = km + (i * sequence_interval)
                if ts < 0 or ts >= dur:
                    continue
                seq_path = os.path.join(output_dir, f'{base}_seq{i:+d}.jpg')
                if _extract_frame(clip.path, ts, seq_path):
                    frames.append(ExtractedFrame(
                        image_path=seq_path,
                        timestamp=ts,
                        label='key_moment' if i == 0 else 'near_key_moment',
                        clip_path=clip.path,
                        streamer=clip.streamer,
                        set_key=clip.set_key,
                        clip_type=clip.clip_type,
                        key_moment_offset=float(i) * sequence_interval,
                    ))

    return frames


@NodeRegistry.register
class FrameExtractorNode(BaseNode):
    node_type = 'frame_extractor'
    label = 'Frame Extractor'
    category = 'video'
    description = 'Extract positive and negative frames from clips at key_moment timestamps'
    inputs = [
        Port('clips', PortType.VIDEO, 'Clips with key_moment metadata'),
    ]
    outputs = [
        Port('frames', PortType.FRAMES, 'Extracted frames with labels'),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'output_dir': {
                'type': 'string',
                'title': 'Frame Output Directory',
                'default': './output/frames',
            },
            'negatives_per_clip': {
                'type': 'integer',
                'title': 'Negatives per Clip',
                'description': 'How many non-key-moment frames to extract per clip',
                'default': 3, 'minimum': 0, 'maximum': 20,
            },
            'sequence_frames': {
                'type': 'integer',
                'title': 'Sequence Frames',
                'description': 'Frames before/after key moment (0 = disabled)',
                'default': 0, 'minimum': 0, 'maximum': 30,
            },
            'sequence_interval': {
                'type': 'number',
                'title': 'Sequence Interval (seconds)',
                'default': 1.0, 'minimum': 0.1,
            },
            'quality': {
                'type': 'integer',
                'title': 'JPEG Quality (2=best, 31=worst)',
                'default': 2, 'minimum': 1, 'maximum': 31,
            },
        },
    }

    def process(self, inputs, config):
        clips = inputs.get('clips', [])
        output_dir = config.get('output_dir', './output/frames')
        negatives = config.get('negatives_per_clip', 3)
        seq_frames = config.get('sequence_frames', 0)
        seq_interval = config.get('sequence_interval', 1.0)

        frames = extract_training_frames(
            clips, output_dir,
            negatives_per_clip=negatives,
            sequence_frames=seq_frames,
            sequence_interval=seq_interval,
        )

        pos_count = sum(1 for f in frames if f.label == 'key_moment')
        neg_count = sum(1 for f in frames if f.label == 'not_key_moment')

        return {
            'frames': frames,
            'metrics': {
                'total_frames': len(frames),
                'positive_frames': pos_count,
                'negative_frames': neg_count,
                'sequence_frames': len(frames) - pos_count - neg_count,
                'clips_processed': len(set(f.clip_path for f in frames)),
            },
        }
