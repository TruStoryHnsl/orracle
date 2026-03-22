"""Region of Interest Extractor — crops frames around detected body zones.

Uses pose keypoint data to identify the active region of the frame
(wrist-hip zone) and extracts a focused crop. This gives the visual
classifier a high-resolution view of only the area that matters,
reducing noise from irrelevant parts of the frame (background, face, etc.)

The ROI is determined by:
  1. Find visible wrists and hips in the pose data
  2. Compute the bounding box around those keypoints
  3. Expand the box by a configurable padding factor
  4. Crop the frame to that region
  5. If no wrists/hips visible, fall back to the lower-center of the frame
     (statistically the most likely activity zone in webcam streams)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from nodes.video.frame_extractor import ExtractedFrame


# COCO keypoint indices for wrists and hips
WRIST_INDICES = [9, 10]   # left_wrist, right_wrist
HIP_INDICES = [11, 12]    # left_hip, right_hip
# Also useful: elbows for arm tracking
ELBOW_INDICES = [7, 8]    # left_elbow, right_elbow


@dataclass
class ROICrop:
    """A cropped region of interest from a frame."""
    crop_path: str              # Path to the cropped image
    source_frame: str           # Original full frame path
    roi_box: list[float]        # [x1, y1, x2, y2] normalized in original image
    roi_source: str             # 'pose' or 'fallback'
    label: str
    streamer: str
    set_key: str
    wrists_visible: int         # How many wrists were detected
    hips_visible: int           # How many hips were detected

    def to_dict(self) -> dict:
        return {
            'crop_path': self.crop_path,
            'source_frame': self.source_frame,
            'roi_box': self.roi_box,
            'roi_source': self.roi_source,
            'label': self.label,
            'streamer': self.streamer,
            'set_key': self.set_key,
            'wrists_visible': self.wrists_visible,
            'hips_visible': self.hips_visible,
        }


def _compute_roi(pose_data: dict, padding: float = 0.15) -> tuple[list[float], str, int, int]:
    """Compute the ROI bounding box from pose keypoints.

    Focuses on wrists and hips. Falls back to lower-center if not visible.

    Args:
        pose_data: Dict with 'landmarks' list from pose extractor
        padding: How much to expand the ROI (fraction of image size)

    Returns:
        (roi_box [x1,y1,x2,y2], source, wrists_visible, hips_visible)
    """
    landmarks = pose_data.get('landmarks', [])
    if not landmarks:
        return [0.2, 0.4, 0.8, 1.0], 'fallback', 0, 0

    # Collect visible wrist and hip positions
    points = []
    wrists_vis = 0
    hips_vis = 0
    min_conf = 0.2

    for idx in WRIST_INDICES:
        if idx < len(landmarks) and landmarks[idx]['visibility'] > min_conf:
            points.append((landmarks[idx]['x'], landmarks[idx]['y']))
            wrists_vis += 1

    for idx in HIP_INDICES:
        if idx < len(landmarks) and landmarks[idx]['visibility'] > min_conf:
            points.append((landmarks[idx]['x'], landmarks[idx]['y']))
            hips_vis += 1

    # Add elbows if we have few points (helps define the active zone)
    if len(points) < 2:
        for idx in ELBOW_INDICES:
            if idx < len(landmarks) and landmarks[idx]['visibility'] > min_conf:
                points.append((landmarks[idx]['x'], landmarks[idx]['y']))

    if len(points) < 2:
        # Not enough keypoints — use the detection bbox if available,
        # otherwise fall back to lower-center
        bbox = pose_data.get('bbox', [])
        if bbox and len(bbox) == 4:
            # Use lower half of the person's bounding box
            x1, y1, x2, y2 = bbox
            mid_y = (y1 + y2) / 2
            return [x1, mid_y, x2, y2], 'bbox_lower', wrists_vis, hips_vis
        return [0.2, 0.4, 0.8, 1.0], 'fallback', wrists_vis, hips_vis

    # Compute bounding box around detected points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Expand by padding
    w = x2 - x1
    h = y2 - y1
    # Ensure minimum size (at least 20% of frame)
    w = max(w, 0.2)
    h = max(h, 0.2)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half_w = (w / 2) + padding
    half_h = (h / 2) + padding

    roi = [
        max(0, cx - half_w),
        max(0, cy - half_h),
        min(1, cx + half_w),
        min(1, cy + half_h),
    ]

    return roi, 'pose', wrists_vis, hips_vis


def extract_roi_crops(
    frames: list[ExtractedFrame],
    pose_dir: str,
    output_dir: str,
    padding: float = 0.15,
    min_crop_size: int = 64,
) -> list[ROICrop]:
    """Extract ROI crops from frames using their associated pose data.

    For each frame:
      1. Load pose JSON sidecar
      2. Compute ROI around wrist-hip zone
      3. Crop the frame to the ROI
      4. Save the crop

    Args:
        frames: ExtractedFrame objects with image_path
        pose_dir: Directory containing pose JSON files
        output_dir: Where to save cropped images
        padding: ROI expansion factor
        min_crop_size: Minimum crop dimension in pixels
    """
    try:
        from PIL import Image
    except ImportError:
        return []

    os.makedirs(output_dir, exist_ok=True)
    crops = []

    for frame in frames:
        # Load pose data
        pose_name = os.path.basename(frame.image_path).rsplit('.', 1)[0] + '_pose.json'
        pose_path = os.path.join(pose_dir, pose_name)

        if os.path.exists(pose_path):
            with open(pose_path) as f:
                pose_data = json.load(f)
        else:
            pose_data = {}

        # Compute ROI
        roi, source, wrists, hips = _compute_roi(pose_data, padding)

        # Crop the frame
        try:
            img = Image.open(frame.image_path)
            w, h = img.size
            crop_box = (
                int(roi[0] * w),
                int(roi[1] * h),
                int(roi[2] * w),
                int(roi[3] * h),
            )
            # Ensure minimum size
            cw = crop_box[2] - crop_box[0]
            ch = crop_box[3] - crop_box[1]
            if cw < min_crop_size or ch < min_crop_size:
                # Too small — use fallback region
                roi = [0.1, 0.3, 0.9, 1.0]
                crop_box = (int(0.1*w), int(0.3*h), int(0.9*w), h)

            cropped = img.crop(crop_box)
            crop_name = os.path.basename(frame.image_path).rsplit('.', 1)[0] + '_roi.jpg'
            crop_path = os.path.join(output_dir, crop_name)
            cropped.save(crop_path, quality=90)

            crops.append(ROICrop(
                crop_path=crop_path,
                source_frame=frame.image_path,
                roi_box=roi,
                roi_source=source,
                label=frame.label,
                streamer=frame.streamer,
                set_key=frame.set_key,
                wrists_visible=wrists,
                hips_visible=hips,
            ))
        except Exception:
            continue

    return crops
