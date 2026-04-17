"""Clip Scanner — discovers clips and reads key_moment metadata.

Scans cbclips directories (short/medium/long), reads the key_moment
timestamp from each MP4's comment metadata tag, and groups clips into
clip sets by their shared basename (YYMMDD-HHMMSS-streamer_id).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field

from nodes.base import BaseNode, NodeRegistry, Port, PortType


@dataclass
class ClipInfo:
    """Metadata for a single clip file."""
    path: str
    filename: str
    streamer: str
    date: str                    # YYMMDD
    time: str                    # HHMMSS
    clip_type: int               # 1=long, 2=medium, 3=short
    duration: float = 0.0        # seconds
    key_moment: float | None = None  # seconds into clip, or None
    set_key: str = ''            # YYMMDD-HHMMSS-streamer (groups clips)

    def to_dict(self) -> dict:
        return {
            'path': self.path,
            'filename': self.filename,
            'streamer': self.streamer,
            'date': self.date,
            'time': self.time,
            'clip_type': self.clip_type,
            'duration': self.duration,
            'key_moment': self.key_moment,
            'set_key': self.set_key,
        }


# Regex for current naming: YYMMDD-HHMMSS-streamer_id-N.mp4
_CLIP_RE = re.compile(r'^(\d{6})-(\d{6})-(.+)-([123])\.mp4$')

# Before-offsets per clip type (must match cbtrimweb trimmer.py)
_BEFORE_OFFSET = {1: 900, 2: 120, 3: 10}  # long, medium, short


def _parse_key_moment(comment: str) -> float | None:
    """Parse key_moment=HH:MM:SS from comment tag into seconds."""
    m = re.search(r'key_moment=(\d{2}):(\d{2}):(\d{2})', comment)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return h * 3600 + mi * 60 + s
    return None


def _derive_key_moment(clip_type: int, time_str: str, duration: float) -> float | None:
    """Derive key_moment from clip structure when no explicit metadata exists.

    Every clip is trimmed around a source timestamp (the HHMMSS in the filename).
    The key moment in the clip is always at the before_offset position, clamped
    to the actual clip duration when duration is known.
    """
    before = _BEFORE_OFFSET.get(clip_type)
    if before is None:
        return None
    # Source timestamp in the daily recording
    try:
        hs, ms, ss = int(time_str[0:2]), int(time_str[2:4]), int(time_str[4:6])
        source_ts = hs * 3600 + ms * 60 + ss
    except (ValueError, IndexError):
        source_ts = before + 1  # assume clip is not near start
    km = min(source_ts, before)
    # Clamp to duration only if we have a reliable duration
    if duration > 2:
        km = min(km, duration - 1)
    return float(km)


def _probe_clip(path: str, timeout: int = 10) -> dict:
    """Read duration and metadata from an MP4 using ffprobe."""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_format', path],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode != 0:
            return {}
        data = json.loads(r.stdout)
        fmt = data.get('format', {})
        return {
            'duration': float(fmt.get('duration', 0)),
            'comment': fmt.get('tags', {}).get('comment', ''),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return {}


def scan_clips(clip_dir: str, clip_types: list[str] | None = None,
               probe_metadata: bool = True) -> list[ClipInfo]:
    """Scan clip directories and return ClipInfo for each valid clip.

    Args:
        clip_dir: Root cbclips directory (contains short/, medium/, long/)
        clip_types: Which subdirs to scan. Default: ['short', 'medium', 'long']
        probe_metadata: If True, read duration + key_moment via ffprobe (slower)
    """
    if clip_types is None:
        clip_types = ['short', 'medium', 'long']

    type_map = {'long': 1, 'medium': 2, 'short': 3}
    clips = []

    for subdir in clip_types:
        type_num = type_map.get(subdir, 0)
        dir_path = os.path.join(clip_dir, subdir)
        if not os.path.isdir(dir_path):
            continue

        try:
            names = os.listdir(dir_path)
        except OSError:
            continue

        for name in sorted(names):
            if not name.endswith('.mp4') or '.meta.' in name:
                continue

            m = _CLIP_RE.match(name)
            if not m:
                continue

            date, time_str, streamer, ctype = m.groups()
            clip_path = os.path.join(dir_path, name)
            set_key = f'{date}-{time_str}-{streamer}'

            clip = ClipInfo(
                path=clip_path,
                filename=name,
                streamer=streamer,
                date=date,
                time=time_str,
                clip_type=int(ctype),
                set_key=set_key,
            )

            if probe_metadata:
                probe = _probe_clip(clip_path)
                clip.duration = probe.get('duration', 0)
                comment = probe.get('comment', '')
                explicit = _parse_key_moment(comment)
                clip.key_moment = explicit if explicit is not None else \
                    _derive_key_moment(int(ctype), time_str, clip.duration)

            clips.append(clip)

    return clips


def group_clip_sets(clips: list[ClipInfo]) -> dict[str, list[ClipInfo]]:
    """Group clips by set_key (same key moment, different durations)."""
    sets: dict[str, list[ClipInfo]] = {}
    for clip in clips:
        sets.setdefault(clip.set_key, []).append(clip)
    return sets


@NodeRegistry.register
class ClipScannerNode(BaseNode):
    node_type = 'clip_scanner'
    label = 'Clip Scanner'
    category = 'video'
    description = 'Scan cbclips directory for clips with key_moment metadata'
    inputs = []
    outputs = [
        Port('clips', PortType.VIDEO, 'List of discovered clips with metadata'),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'clip_dir': {
                'type': 'string',
                'title': 'Clips Directory',
                'description': 'Root cbclips directory',
                'default': '/mnt/vault/watch/cbclips',
            },
            'clip_types': {
                'type': 'string',
                'title': 'Clip Types',
                'description': 'Comma-separated: short, medium, long (empty = all)',
                'default': 'short',
            },
            'probe_metadata': {
                'type': 'boolean',
                'title': 'Read Metadata',
                'description': 'Read duration + key_moment from each clip (slower)',
                'default': True,
            },
        },
        'required': ['clip_dir'],
    }

    def process(self, inputs, config):
        clip_dir = config.get('clip_dir', '/mnt/vault/watch/cbclips')
        types_str = config.get('clip_types', 'short').strip()
        clip_types = [t.strip() for t in types_str.split(',') if t.strip()] or None
        probe = config.get('probe_metadata', True)

        clips = scan_clips(clip_dir, clip_types, probe)
        return {
            'clips': clips,
            'metrics': {
                'total_clips': len(clips),
                'with_key_moment': sum(1 for c in clips if c.key_moment is not None),
                'streamers': len(set(c.streamer for c in clips)),
            },
        }
