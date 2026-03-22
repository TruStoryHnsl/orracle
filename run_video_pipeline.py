#!/usr/bin/env python3
"""Run the video key-moment training data pipeline.

Scans cbclips directories for clips with key_moment metadata,
extracts positive/negative frames, and exports training pairs.

Can run as a one-shot batch or in watch mode for autonomous operation.

Usage:
    python run_video_pipeline.py                    # One-shot batch
    python run_video_pipeline.py --watch            # Watch mode (continuous)
    python run_video_pipeline.py --watch --interval 300  # Check every 5 min
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
VENV_PYTHON = SCRIPT_DIR / 'venv' / 'bin' / 'python'
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

from nodes.video.clip_scanner import scan_clips, ClipInfo
from nodes.video.frame_extractor import extract_training_frames
from nodes.video.audio_extractor import extract_audio_segments
from nodes.video.pose_extractor import extract_pose_data
from nodes.video.roi_extractor import extract_roi_crops
from nodes.video.motion_analyzer import analyze_motion
from nodes.video.training_pairs import export_jsonl, export_csv, train_val_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLIP_DIR = os.environ.get('ORRACLE_CLIPS', '/mnt/vault/watch/cbclips')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', 'video_training')
FRAMES_DIR = os.path.join(OUTPUT_DIR, 'frames')
ROI_DIR = os.path.join(OUTPUT_DIR, 'roi')
AUDIO_DIR = os.path.join(OUTPUT_DIR, 'audio')
POSE_DIR = os.path.join(OUTPUT_DIR, 'pose')
MOTION_DIR = os.path.join(OUTPUT_DIR, 'motion')
PROCESSED_LOG = os.path.join(OUTPUT_DIR, '.processed.json')


def log(msg: str):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def load_processed() -> set[str]:
    """Load set of already-processed clip set_keys."""
    if os.path.exists(PROCESSED_LOG):
        try:
            with open(PROCESSED_LOG) as f:
                return set(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return set()


def save_processed(processed: set[str]):
    """Save processed set_keys to disk."""
    os.makedirs(os.path.dirname(PROCESSED_LOG), exist_ok=True)
    with open(PROCESSED_LOG, 'w') as f:
        json.dump(sorted(processed), f)


def run_batch(clip_types: list[str], negatives: int, sequence_frames: int,
              sequence_interval: float, val_ratio: float,
              incremental: bool = True) -> dict:
    """Run a single batch: scan → extract → export.

    If incremental, skips clips that have already been processed.
    """
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Phase 1: Scan
    log(f'Scanning {CLIP_DIR} for clips ({", ".join(clip_types)})...')
    all_clips = scan_clips(CLIP_DIR, clip_types, probe_metadata=True)
    labeled = [c for c in all_clips if c.key_moment is not None]
    log(f'  Found {len(all_clips)} clips, {len(labeled)} with key_moment metadata')

    if not labeled:
        log('No labeled clips found.')
        return {'status': 'empty', 'total_clips': len(all_clips)}

    # Filter already-processed clips (incremental mode)
    if incremental:
        processed = load_processed()
        new_clips = [c for c in labeled if c.set_key not in processed]
        log(f'  {len(new_clips)} new clips ({len(processed)} already processed)')
        if not new_clips:
            log('Nothing new to process.')
            return {'status': 'up_to_date', 'processed': len(processed)}
    else:
        new_clips = labeled
        processed = set()

    # Phase 2: Extract frames
    log(f'Extracting frames from {len(new_clips)} clips '
        f'({negatives} negatives/clip, seq={sequence_frames})...')
    new_frames = extract_training_frames(
        new_clips, FRAMES_DIR,
        negatives_per_clip=negatives,
        sequence_frames=sequence_frames,
        sequence_interval=sequence_interval,
    )
    pos = sum(1 for f in new_frames if f.label == 'key_moment')
    neg = sum(1 for f in new_frames if f.label == 'not_key_moment')
    seq = len(new_frames) - pos - neg
    log(f'  Extracted {len(new_frames)} frames '
        f'({pos} positive, {neg} negative, {seq} sequence)')

    # Phase 2b: Extract audio segments
    os.makedirs(AUDIO_DIR, exist_ok=True)
    log(f'Extracting audio segments (5s windows, spectrograms)...')
    new_audio = extract_audio_segments(
        new_clips, AUDIO_DIR,
        segment_duration=5.0,
        negatives_per_clip=2,
        generate_spectrograms=True,
    )
    audio_pos = sum(1 for a in new_audio if a.label == 'key_moment')
    audio_neg = sum(1 for a in new_audio if a.label == 'not_key_moment')
    log(f'  Extracted {len(new_audio)} audio segments '
        f'({audio_pos} positive, {audio_neg} negative)')

    # Phase 2c: Pose detection on extracted frames
    os.makedirs(POSE_DIR, exist_ok=True)
    log(f'Running pose detection on {len(new_frames)} frames...')
    pose_results = extract_pose_data(new_frames, POSE_DIR)
    persons_found = sum(1 for p in pose_results if p.has_person)
    log(f'  {persons_found}/{len(pose_results)} frames have detected pose')

    # Phase 2d: ROI extraction (crop frames to wrist-hip zone)
    os.makedirs(ROI_DIR, exist_ok=True)
    log(f'Extracting ROI crops (wrist-hip focus)...')
    roi_crops = extract_roi_crops(new_frames, POSE_DIR, ROI_DIR, padding=0.15)
    pose_rois = sum(1 for r in roi_crops if r.roi_source == 'pose')
    bbox_rois = sum(1 for r in roi_crops if r.roi_source == 'bbox_lower')
    fallback_rois = sum(1 for r in roi_crops if r.roi_source == 'fallback')
    log(f'  {len(roi_crops)} ROI crops ({pose_rois} from pose, {bbox_rois} from bbox, {fallback_rois} fallback)')

    # Phase 2e: Motion analysis around key moments
    os.makedirs(MOTION_DIR, exist_ok=True)
    log(f'Analyzing motion patterns (10s window, 2fps)...')
    motion_profiles = analyze_motion(
        new_clips, MOTION_DIR,
        window=10.0,
        fps=2.0,
    )
    if motion_profiles:
        avg_ratio = sum(m.motion_ratio for m in motion_profiles) / len(motion_profiles)
        log(f'  {len(motion_profiles)} motion profiles, avg km/baseline ratio: {avg_ratio:.2f}x')
    else:
        log(f'  No motion profiles generated')

    # Mark as processed
    for c in new_clips:
        processed.add(c.set_key)
    save_processed(processed)

    # Phase 3: Rebuild full training set from all frames on disk
    log('Rebuilding training set from all extracted frames...')
    all_frames_on_disk = _scan_frames_dir(FRAMES_DIR)
    log(f'  {len(all_frames_on_disk)} total frames on disk')

    # Split and export
    train_frames, val_frames = train_val_split(
        all_frames_on_disk, val_ratio, stratify_by_streamer=True)

    train_result = export_csv(train_frames, os.path.join(OUTPUT_DIR, 'train.csv'))
    val_result = export_csv(val_frames, os.path.join(OUTPUT_DIR, 'val.csv'))
    train_jsonl_result = export_jsonl(
        train_frames, os.path.join(OUTPUT_DIR, 'train.jsonl'),
        include_base64=False)
    val_jsonl_result = export_jsonl(
        val_frames, os.path.join(OUTPUT_DIR, 'val.jsonl'),
        include_base64=False)

    elapsed = time.time() - start

    log(f'\n{"="*50}')
    log(f'BATCH COMPLETE ({elapsed:.0f}s)')
    log(f'{"="*50}')
    log(f'New clips processed: {len(new_clips)}')
    log(f'New frames extracted: {len(new_frames)}')
    log(f'Total frames on disk: {len(all_frames_on_disk)}')
    log(f'Train: {train_result["count"]} frames ({pos} pos, {neg} neg)')
    log(f'Val:   {val_result["count"]} frames')
    log(f'Output: {OUTPUT_DIR}/')

    return {
        'status': 'completed',
        'new_clips': len(new_clips),
        'new_frames': len(new_frames),
        'total_frames': len(all_frames_on_disk),
        'train_count': train_result['count'],
        'val_count': val_result['count'],
        'elapsed': elapsed,
    }


def _scan_frames_dir(frames_dir: str) -> list:
    """Rebuild frame list from filenames on disk.

    Frame naming convention:
      {set_key}-{clip_type}_km.jpg       → key_moment
      {set_key}-{clip_type}_neg{N}.jpg   → not_key_moment
      {set_key}-{clip_type}_seq{N}.jpg   → near_key_moment
    """
    import re
    from nodes.video.frame_extractor import ExtractedFrame

    frames = []
    if not os.path.isdir(frames_dir):
        return frames

    km_re = re.compile(r'^(.+)-(\d)_km\.jpg$')
    neg_re = re.compile(r'^(.+)-(\d)_neg\d+\.jpg$')
    seq_re = re.compile(r'^(.+)-(\d)_seq[+-]\d+\.jpg$')

    for name in os.listdir(frames_dir):
        if not name.endswith('.jpg'):
            continue
        path = os.path.join(frames_dir, name)

        # Determine label from filename
        m = km_re.match(name)
        if m:
            label = 'key_moment'
            set_key, clip_type = m.group(1), int(m.group(2))
        else:
            m = neg_re.match(name)
            if m:
                label = 'not_key_moment'
                set_key, clip_type = m.group(1), int(m.group(2))
            else:
                m = seq_re.match(name)
                if m:
                    label = 'near_key_moment'
                    set_key, clip_type = m.group(1), int(m.group(2))
                else:
                    continue

        # Extract streamer from set_key (YYMMDD-HHMMSS-streamer)
        parts = set_key.split('-', 2)
        streamer = parts[2] if len(parts) > 2 else 'unknown'

        frames.append(ExtractedFrame(
            image_path=path,
            timestamp=0,  # Not available from filename alone
            label=label,
            clip_path='',
            streamer=streamer,
            set_key=set_key,
            clip_type=clip_type,
            key_moment_offset=0,
        ))

    return frames


def watch_mode(args):
    """Continuously watch for new clips and process them."""
    log(f'Watch mode: checking every {args.interval}s')
    log(f'Clip dir: {CLIP_DIR}')
    log(f'Output: {OUTPUT_DIR}')

    while True:
        try:
            result = run_batch(
                clip_types=args.clip_types.split(','),
                negatives=args.negatives,
                sequence_frames=args.sequence,
                sequence_interval=args.sequence_interval,
                val_ratio=args.val_ratio,
                incremental=True,
            )
            if result['status'] == 'completed':
                log(f'Sleeping {args.interval}s until next check...')
            elif result['status'] == 'up_to_date':
                log(f'No new clips. Sleeping {args.interval}s...')
        except Exception as e:
            log(f'ERROR: {e}')
            log(f'Retrying in {args.interval}s...')

        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(
        description='Video key-moment training data pipeline')
    parser.add_argument('--watch', action='store_true',
                        help='Run continuously, processing new clips')
    parser.add_argument('--interval', type=int, default=300,
                        help='Watch mode check interval in seconds (default: 300)')
    parser.add_argument('--clip-types', default='short',
                        help='Clip types to process (comma-separated, default: short)')
    parser.add_argument('--negatives', type=int, default=3,
                        help='Negative frames per clip (default: 3)')
    parser.add_argument('--sequence', type=int, default=0,
                        help='Sequence frames before/after key moment (default: 0)')
    parser.add_argument('--sequence-interval', type=float, default=1.0,
                        help='Seconds between sequence frames (default: 1.0)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--full', action='store_true',
                        help='Reprocess all clips (ignore processed log)')
    args = parser.parse_args()

    if args.watch:
        watch_mode(args)
    else:
        run_batch(
            clip_types=args.clip_types.split(','),
            negatives=args.negatives,
            sequence_frames=args.sequence,
            sequence_interval=args.sequence_interval,
            val_ratio=args.val_ratio,
            incremental=not args.full,
        )


if __name__ == '__main__':
    main()
