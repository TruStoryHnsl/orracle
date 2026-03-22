"""Audio Extractor — extracts audio segments from clips around key moments.

For each clip with a key_moment, extracts:
  - A short audio segment centered on the key moment (positive)
  - Short audio segments at random offsets (negatives)
  - Generates mel spectrograms as images for visual model training

Uses ffmpeg for audio extraction and numpy/scipy for spectrogram generation.
Falls back to raw WAV export if scipy is not available.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from nodes.video.clip_scanner import ClipInfo


@dataclass
class AudioSegment:
    """An extracted audio segment with its label and metadata."""
    wav_path: str                # Path to the WAV file
    spectrogram_path: str | None # Path to the spectrogram image (if generated)
    timestamp: float             # Center timestamp in the clip
    duration: float              # Segment duration in seconds
    label: str                   # 'key_moment' or 'not_key_moment'
    clip_path: str
    streamer: str
    set_key: str
    clip_type: int
    key_moment_offset: float

    def to_dict(self) -> dict:
        return {
            'wav_path': self.wav_path,
            'spectrogram_path': self.spectrogram_path,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'label': self.label,
            'clip_path': self.clip_path,
            'streamer': self.streamer,
            'set_key': self.set_key,
            'clip_type': self.clip_type,
            'key_moment_offset': self.key_moment_offset,
        }


def _extract_audio_segment(clip_path: str, timestamp: float, duration: float,
                           output_path: str, sample_rate: int = 16000,
                           timeout: int = 15) -> bool:
    """Extract a mono WAV audio segment from a video clip.

    Args:
        clip_path: Path to MP4 file
        timestamp: Center of the segment in seconds
        duration: Length of segment in seconds
        output_path: Where to save the WAV
        sample_rate: Audio sample rate (16kHz is standard for speech/audio ML)
        timeout: ffmpeg timeout
    """
    start = max(0, timestamp - duration / 2)
    try:
        r = subprocess.run(
            ['ffmpeg', '-y',
             '-ss', str(start),
             '-i', clip_path,
             '-t', str(duration),
             '-vn',                    # No video
             '-ar', str(sample_rate),  # Resample
             '-ac', '1',              # Mono
             '-f', 'wav',
             output_path],
            capture_output=True, timeout=timeout,
        )
        return r.returncode == 0 and os.path.exists(output_path)
    except subprocess.TimeoutExpired:
        return False


def _generate_spectrogram(wav_path: str, output_path: str,
                          n_fft: int = 1024, hop_length: int = 256) -> bool:
    """Generate a mel spectrogram image from a WAV file.

    Uses scipy + numpy if available. Returns False if dependencies missing.
    """
    try:
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import stft
    except ImportError:
        return False

    try:
        rate, data = wavfile.read(wav_path)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / max(abs(data.min()), abs(data.max()), 1)

        # Compute STFT
        _, _, Zxx = stft(data, fs=rate, nperseg=n_fft, noverlap=n_fft - hop_length)
        magnitude = np.abs(Zxx)

        # Log scale
        log_spec = np.log1p(magnitude * 100)

        # Normalize to 0-255 for image output
        if log_spec.max() > 0:
            log_spec = (log_spec / log_spec.max() * 255).astype(np.uint8)
        else:
            log_spec = np.zeros_like(log_spec, dtype=np.uint8)

        # Flip vertically (low freq at bottom) and save as grayscale image
        log_spec = np.flipud(log_spec)

        # Use PIL if available, otherwise save raw pgm
        try:
            from PIL import Image
            img = Image.fromarray(log_spec, mode='L')
            img.save(output_path)
            return True
        except ImportError:
            pass

        # Fallback: save as PGM (portable graymap)
        h, w = log_spec.shape
        pgm_path = output_path.rsplit('.', 1)[0] + '.pgm'
        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{w} {h}\n255\n'.encode())
            f.write(log_spec.tobytes())
        return True

    except Exception:
        return False


def extract_audio_segments(
    clips: list[ClipInfo],
    output_dir: str,
    segment_duration: float = 5.0,
    negatives_per_clip: int = 2,
    generate_spectrograms: bool = True,
    seed: int = 42,
) -> list[AudioSegment]:
    """Extract positive and negative audio segments from clips.

    For each clip with a key_moment:
      - Extract audio centered on the key_moment (positive)
      - Extract audio at random offsets (negatives)
      - Optionally generate spectrogram images

    Args:
        clips: List of ClipInfo with key_moment metadata
        output_dir: Directory to save WAV files and spectrograms
        segment_duration: Length of each audio segment in seconds
        negatives_per_clip: How many negative segments per clip
        generate_spectrograms: Whether to generate spectrogram images
        seed: Random seed

    Returns:
        List of AudioSegment objects
    """
    import random
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    segments = []

    labeled = [c for c in clips if c.key_moment is not None and c.duration > 0]

    for clip in labeled:
        km = clip.key_moment
        dur = clip.duration
        base = f'{clip.set_key}-{clip.clip_type}'
        half = segment_duration / 2

        # Positive: audio at key moment
        pos_wav = os.path.join(output_dir, f'{base}_km.wav')
        if _extract_audio_segment(clip.path, km, segment_duration, pos_wav):
            spec_path = None
            if generate_spectrograms:
                spec_img = os.path.join(output_dir, f'{base}_km_spec.png')
                if _generate_spectrogram(pos_wav, spec_img):
                    spec_path = spec_img

            segments.append(AudioSegment(
                wav_path=pos_wav,
                spectrogram_path=spec_path,
                timestamp=km,
                duration=segment_duration,
                label='key_moment',
                clip_path=clip.path,
                streamer=clip.streamer,
                set_key=clip.set_key,
                clip_type=clip.clip_type,
                key_moment_offset=0.0,
            ))

        # Negatives
        margin = segment_duration + 2.0
        for neg_idx in range(negatives_per_clip):
            # Pick a random timestamp away from key moment
            candidates = []
            if km - margin > half:
                candidates.append((half, km - margin))
            if km + margin < dur - half:
                candidates.append((km + margin, dur - half))

            if not candidates:
                continue

            low, high = rng.choice(candidates)
            if high <= low:
                continue
            ts = rng.uniform(low, high)

            neg_wav = os.path.join(output_dir, f'{base}_neg{neg_idx}.wav')
            if _extract_audio_segment(clip.path, ts, segment_duration, neg_wav):
                spec_path = None
                if generate_spectrograms:
                    spec_img = os.path.join(output_dir, f'{base}_neg{neg_idx}_spec.png')
                    if _generate_spectrogram(neg_wav, spec_img):
                        spec_path = spec_img

                segments.append(AudioSegment(
                    wav_path=neg_wav,
                    spectrogram_path=spec_path,
                    timestamp=ts,
                    duration=segment_duration,
                    label='not_key_moment',
                    clip_path=clip.path,
                    streamer=clip.streamer,
                    set_key=clip.set_key,
                    clip_type=clip.clip_type,
                    key_moment_offset=ts - km,
                ))

    return segments
