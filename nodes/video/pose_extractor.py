"""Pose Extractor — detects body keypoints and bounding boxes in frames.

Uses YOLOv8-pose for robust person detection and keypoint estimation.
Handles partial bodies, close-ups, and unusual angles — significantly
better than MediaPipe for webcam stream content.

COCO keypoints (17 points):
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

Falls back gracefully if ultralytics is not installed.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from nodes.video.frame_extractor import ExtractedFrame

# Try importing YOLOv8
_HAS_YOLO = False
_yolo_model = None
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    pass

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# COCO skeleton connections
SKELETON_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4],       # face
    [5, 6],                                  # shoulders
    [5, 7], [7, 9],                          # left arm
    [6, 8], [8, 10],                         # right arm
    [5, 11], [6, 12],                        # torso
    [11, 12],                                # hips
    [11, 13], [13, 15],                      # left leg
    [12, 14], [14, 16],                      # right leg
]


def _get_model():
    """Lazy-load the YOLO model (downloads on first use)."""
    global _yolo_model
    if _yolo_model is None and _HAS_YOLO:
        _yolo_model = YOLO('yolov8n-pose.pt')
    return _yolo_model


@dataclass
class PoseData:
    """Keypoint and bounding box data for a single frame."""
    frame_path: str
    keypoints_path: str
    label: str
    streamer: str
    set_key: str
    num_landmarks: int
    visible_landmarks: int
    confidence: float            # Person detection confidence
    has_person: bool
    bbox: list[float]            # [x1, y1, x2, y2] normalized
    pose_magnitude: float

    def to_dict(self) -> dict:
        return {
            'frame_path': self.frame_path,
            'keypoints_path': self.keypoints_path,
            'label': self.label,
            'streamer': self.streamer,
            'set_key': self.set_key,
            'num_landmarks': self.num_landmarks,
            'visible_landmarks': self.visible_landmarks,
            'confidence': self.confidence,
            'has_person': self.has_person,
            'bbox': self.bbox,
            'pose_magnitude': self.pose_magnitude,
        }


def _detect_pose_in_image(image_path: str) -> dict | None:
    """Run YOLOv8-pose on a single image.

    Returns dict with landmarks, bbox, confidence — or None if no person found.
    """
    if not _HAS_YOLO:
        return None

    model = _get_model()
    if model is None:
        return None

    try:
        results = model(image_path, verbose=False)
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return None

        # Use the highest-confidence person
        best_idx = int(r.boxes.conf.argmax())
        conf = float(r.boxes.conf[best_idx])

        # Bounding box (normalized)
        box = r.boxes.xyxyn[best_idx].tolist()

        # Keypoints
        landmarks = []
        visible_count = 0
        if r.keypoints is not None and r.keypoints.data is not None:
            kp = r.keypoints.data[best_idx]  # [17, 3] tensor: x, y, conf
            img_h, img_w = r.orig_shape
            for i in range(kp.shape[0]):
                x = float(kp[i, 0]) / img_w  # normalize to 0-1
                y = float(kp[i, 1]) / img_h
                v = float(kp[i, 2])
                landmarks.append({
                    'x': round(x, 4),
                    'y': round(y, 4),
                    'z': 0.0,
                    'visibility': round(v, 4),
                })
                if v > 0.3:
                    visible_count += 1

        return {
            'landmarks': landmarks,
            'bbox': [round(b, 4) for b in box],
            'confidence': round(conf, 4),
            'visible_landmarks': visible_count,
            'num_persons': len(r.boxes),
        }

    except Exception as e:
        import sys
        print(f'[pose] Error processing {image_path}: {e}', file=sys.stderr)
        return None


def _compute_pose_features(landmarks: list[dict]) -> dict:
    """Compute aggregate features from landmarks."""
    if not landmarks:
        return {'confidence': 0, 'has_person': False, 'magnitude': 0}

    visibilities = [lm['visibility'] for lm in landmarks]
    avg_conf = sum(visibilities) / len(visibilities)
    has_person = any(v > 0.3 for v in visibilities)

    xs = [lm['x'] for lm in landmarks if lm['visibility'] > 0.3]
    ys = [lm['y'] for lm in landmarks if lm['visibility'] > 0.3]
    if xs and ys:
        magnitude = (max(xs) - min(xs)) * (max(ys) - min(ys))
    else:
        magnitude = 0

    return {
        'confidence': round(avg_conf, 4),
        'has_person': has_person,
        'magnitude': round(magnitude, 4),
    }


def compute_velocity(kp_before: list[dict], kp_after: list[dict],
                     dt: float = 1.0) -> list[dict]:
    """Compute per-landmark velocity between two frames."""
    if not kp_before or not kp_after or len(kp_before) != len(kp_after):
        return []

    velocities = []
    for i, (a, b) in enumerate(zip(kp_before, kp_after)):
        if a['visibility'] < 0.3 or b['visibility'] < 0.3:
            continue
        vx = (b['x'] - a['x']) / dt
        vy = (b['y'] - a['y']) / dt
        speed = math.sqrt(vx**2 + vy**2)
        velocities.append({
            'landmark': i,
            'name': KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f'kp_{i}',
            'vx': round(vx, 4),
            'vy': round(vy, 4),
            'speed': round(speed, 4),
        })

    return velocities


def extract_pose_data(
    frames: list[ExtractedFrame],
    output_dir: str | None = None,
) -> list[PoseData]:
    """Run pose detection on extracted frames and save keypoint JSON sidecars."""
    results = []

    for frame in frames:
        kp_path = frame.image_path.rsplit('.', 1)[0] + '_pose.json'
        if output_dir:
            kp_path = os.path.join(output_dir, os.path.basename(kp_path))

        pose = _detect_pose_in_image(frame.image_path)

        if pose:
            landmarks = pose['landmarks']
            features = _compute_pose_features(landmarks)
            data = {
                'landmarks': landmarks,
                'bbox': pose['bbox'],
                'confidence': pose['confidence'],
                'visible_landmarks': pose['visible_landmarks'],
                'num_persons': pose['num_persons'],
                'features': features,
                'label': frame.label,
                'set_key': frame.set_key,
                'timestamp': frame.timestamp,
                'key_moment_offset': frame.key_moment_offset,
            }
        else:
            landmarks = []
            features = {'confidence': 0, 'has_person': False, 'magnitude': 0}
            data = {
                'landmarks': [],
                'bbox': [],
                'confidence': 0,
                'visible_landmarks': 0,
                'num_persons': 0,
                'features': features,
                'label': frame.label,
                'set_key': frame.set_key,
            }

        os.makedirs(os.path.dirname(kp_path), exist_ok=True)
        with open(kp_path, 'w') as f:
            json.dump(data, f)

        results.append(PoseData(
            frame_path=frame.image_path,
            keypoints_path=kp_path,
            label=frame.label,
            streamer=frame.streamer,
            set_key=frame.set_key,
            num_landmarks=len(landmarks),
            visible_landmarks=pose['visible_landmarks'] if pose else 0,
            confidence=pose['confidence'] if pose else 0,
            has_person=features['has_person'],
            bbox=pose['bbox'] if pose else [],
            pose_magnitude=features['magnitude'],
        ))

    return results
