"""Motion Analyzer — detects motion intensity between consecutive frames.

Extracts a burst of frames around each key moment and computes:
  - Frame-to-frame pixel difference magnitude
  - Motion intensity over time (activity curve)
  - Peak motion detection (sudden changes)

Uses ffmpeg for frame extraction and numpy for difference computation.
No heavy dependencies (no OpenCV required).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

from nodes.video.clip_scanner import ClipInfo


@dataclass
class MotionProfile:
    """Motion analysis results for a clip around its key moment."""
    profile_path: str            # JSON file with motion curve
    clip_path: str
    streamer: str
    set_key: str
    clip_type: int
    label: str
    key_moment: float
    peak_motion_offset: float    # Seconds from key_moment to peak motion
    avg_motion_at_km: float      # Average motion at key moment
    avg_motion_baseline: float   # Average motion at non-key-moment times
    motion_ratio: float          # km_motion / baseline_motion

    def to_dict(self) -> dict:
        return {
            'profile_path': self.profile_path,
            'clip_path': self.clip_path,
            'streamer': self.streamer,
            'set_key': self.set_key,
            'clip_type': self.clip_type,
            'label': self.label,
            'key_moment': self.key_moment,
            'peak_motion_offset': self.peak_motion_offset,
            'avg_motion_at_km': self.avg_motion_at_km,
            'avg_motion_baseline': self.avg_motion_baseline,
            'motion_ratio': self.motion_ratio,
        }


def _extract_frame_burst(clip_path: str, center: float, window: float,
                         fps: float, output_dir: str, prefix: str,
                         timeout: int = 30) -> list[str]:
    """Extract a burst of frames around a timestamp.

    Returns list of extracted frame paths in chronological order.
    """
    start = max(0, center - window / 2)
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, f'{prefix}_%04d.jpg')

    try:
        r = subprocess.run(
            ['ffmpeg', '-y',
             '-ss', str(start),
             '-i', clip_path,
             '-t', str(window),
             '-vf', f'fps={fps}',
             '-q:v', '5',  # Lower quality for motion analysis (speed)
             pattern],
            capture_output=True, timeout=timeout,
        )
        if r.returncode != 0:
            return []
    except subprocess.TimeoutExpired:
        return []

    # Collect extracted frames in order
    frames = []
    i = 1
    while True:
        path = os.path.join(output_dir, f'{prefix}_{i:04d}.jpg')
        if os.path.exists(path):
            frames.append(path)
            i += 1
        else:
            break

    return frames


def _compute_frame_diff(path_a: str, path_b: str) -> float:
    """Compute normalized pixel difference between two frames.

    Returns a value between 0.0 (identical) and 1.0 (completely different).
    Uses raw pixel comparison without OpenCV.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return 0.0

    try:
        a = np.array(Image.open(path_a).convert('L').resize((160, 120)))
        b = np.array(Image.open(path_b).convert('L').resize((160, 120)))
        diff = np.abs(a.astype(float) - b.astype(float))
        return float(diff.mean() / 255.0)
    except Exception:
        return 0.0


def analyze_motion(
    clips: list[ClipInfo],
    output_dir: str,
    window: float = 10.0,
    fps: float = 2.0,
) -> list[MotionProfile]:
    """Analyze motion patterns around key moments in clips.

    For each clip:
      1. Extract a burst of frames around the key moment (±window/2 seconds)
      2. Compute frame-to-frame differences
      3. Find peak motion and compare to baseline

    Args:
        clips: Clips with key_moment metadata
        output_dir: Directory for motion analysis output
        window: Total analysis window in seconds (centered on key_moment)
        fps: Frame extraction rate for motion analysis
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, '_temp_frames')
    profiles = []

    labeled = [c for c in clips if c.key_moment is not None and c.duration > 0]

    for clip in labeled:
        km = clip.key_moment
        base = f'{clip.set_key}-{clip.clip_type}'

        # Extract frame burst
        frames = _extract_frame_burst(
            clip.path, km, window, fps, temp_dir, base)

        if len(frames) < 3:
            continue

        # Compute frame-to-frame differences
        diffs = []
        for i in range(1, len(frames)):
            d = _compute_frame_diff(frames[i-1], frames[i])
            diffs.append(d)

        if not diffs:
            continue

        # Map diffs to timestamps
        dt = 1.0 / fps
        start_time = max(0, km - window / 2)
        timestamps = [start_time + (i + 0.5) * dt for i in range(len(diffs))]

        # Find which diffs are near the key moment vs baseline
        km_window = 2.0  # ±2 seconds around key moment
        km_diffs = [d for t, d in zip(timestamps, diffs)
                    if abs(t - km) <= km_window]
        baseline_diffs = [d for t, d in zip(timestamps, diffs)
                         if abs(t - km) > km_window]

        avg_km = sum(km_diffs) / len(km_diffs) if km_diffs else 0
        avg_base = sum(baseline_diffs) / len(baseline_diffs) if baseline_diffs else 0.001
        ratio = avg_km / max(avg_base, 0.001)

        # Find peak motion
        peak_idx = diffs.index(max(diffs))
        peak_time = timestamps[peak_idx]
        peak_offset = peak_time - km

        # Save motion profile
        profile_data = {
            'set_key': clip.set_key,
            'clip_type': clip.clip_type,
            'key_moment': km,
            'window': window,
            'fps': fps,
            'motion_curve': [
                {'time': round(t, 2), 'diff': round(d, 4)}
                for t, d in zip(timestamps, diffs)
            ],
            'avg_motion_at_km': round(avg_km, 4),
            'avg_motion_baseline': round(avg_base, 4),
            'motion_ratio': round(ratio, 2),
            'peak_motion_offset': round(peak_offset, 2),
            'peak_motion_value': round(max(diffs), 4),
        }

        profile_path = os.path.join(output_dir, f'{base}_motion.json')
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)

        profiles.append(MotionProfile(
            profile_path=profile_path,
            clip_path=clip.path,
            streamer=clip.streamer,
            set_key=clip.set_key,
            clip_type=clip.clip_type,
            label='key_moment',
            key_moment=km,
            peak_motion_offset=peak_offset,
            avg_motion_at_km=avg_km,
            avg_motion_baseline=avg_base,
            motion_ratio=ratio,
        ))

        # Clean up temp frames for this clip
        for fp in frames:
            try:
                os.remove(fp)
            except OSError:
                pass

    # Clean up temp dir
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    return profiles
