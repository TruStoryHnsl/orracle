#!/usr/bin/env python3
"""Run the full Nifty Archive training data pipeline.

Reads stories from the NAS vault using a thread pool for parallel I/O,
processes through the 135-rule cleaning pipeline, and exports
train.jsonl + val.jsonl for MLX fine-tuning of orrvert_v2.
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
VENV_PYTHON = SCRIPT_DIR / 'venv' / 'bin' / 'python'
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

from nodes.base import DataChunk
from nodes.text.metadata import extract_metadata
from nodes.text.html_strip import strip_html, _is_html
from nodes.text.header_strip import strip_email_headers
from nodes.text.boilerplate import BoilerplateNode
from nodes.text.regex_rules import load_rule_library, apply_rules
from nodes.text.reflow import reflow_text
from nodes.text.dedup import DedupNode
from nodes.text.quality_filter import QualityFilterNode
import nodes  # noqa: ensure registration

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VAULT_DIR = os.environ.get('ORRACLE_SOURCE', '/mnt/vault/read/nifty/gay')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
VAL_RATIO = 0.1
SEED = 42
MIN_CHARS = 500
BATCH_SIZE = 200
IO_WORKERS = 8    # parallel NAS reads (lower = less CPU contention)
MAX_CHUNK_TOKENS = 7000     # Target chunk size (~28K chars)
OVERLAP_TOKENS = 500        # Sliding window overlap for oversized chapters (~2K chars)
CHAPTER_SEP = '\n\n---\n\n'


# ---------------------------------------------------------------------------
# Vault scanning — os.listdir only, no stat calls
# ---------------------------------------------------------------------------

def scan_vault(vault_dir: str) -> list[dict]:
    """Scan the vault for story entries using only os.listdir (no stat)."""
    skip_ext = {'.jpg', '.png', '.gif', '.pdf', '.css', '.js',
                '.mp3', '.mp4', '.zip', '.gz', '.ds_store'}
    entries = []

    try:
        category_names = sorted(os.listdir(vault_dir))
    except OSError as e:
        log(f'  ERROR: Cannot list vault: {e}')
        return []

    for category in category_names:
        if category.startswith('.'):
            continue
        cat_path = os.path.join(vault_dir, category)

        log(f'    Scanning {category}...')
        try:
            names = os.listdir(cat_path)
        except OSError as e:
            log(f'    WARNING: Failed to read {category}: {e}')
            continue

        count = 0
        for name in names:
            if name.startswith('.'):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in skip_ext:
                continue
            entries.append({
                'path': os.path.join(cat_path, name),
                'name': name,
                'category': category,
            })
            count += 1

        log(f'      {count} entries')

    return entries


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def _read_text_file(path: str, encoding: str = 'utf-8') -> str | None:
    """Read a single text file. Skip binary files. No stat calls."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if not data or b'\x00' in data[:512]:
            return None
        return data.decode(encoding, errors='replace')
    except OSError:
        return None


def read_story(path: str, encoding: str = 'utf-8') -> str | None:
    """Read a single story — handles both files and multi-chapter dirs.

    Uses os.listdir + direct open() to avoid stat calls over NAS.
    Tries to open as file first; if that fails with IsADirectoryError,
    treats it as a multi-chapter directory.
    """
    # Try reading as a file first (one open call, no stat)
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if not data or b'\x00' in data[:512]:
            return None
        return data.decode(encoding, errors='replace')
    except IsADirectoryError:
        pass
    except OSError:
        return None

    # It's a directory — multi-chapter story
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return None

    chapters = []
    for name in names:
        if name.startswith('.'):
            continue
        text = _read_text_file(os.path.join(path, name), encoding)
        if text:
            chapters.append(text)

    return '\n\n---\n\n'.join(chapters) if chapters else None


def read_entry(entry: dict) -> tuple[dict, str | None, str | None]:
    """Thread-safe wrapper: read one entry, return (entry, text, error)."""
    text = read_story(entry['path'])
    if text is None:
        # Distinguish binary skip from actual error
        try:
            with open(entry['path'], 'rb') as f:
                sample = f.read(4)
            # File opened fine — it was binary or empty, not a NAS error
            return entry, None, 'skip'
        except IsADirectoryError:
            # Dir that had no readable chapters
            return entry, None, 'empty_dir'
        except OSError as e:
            return entry, None, str(e)
    return entry, text, None


# ---------------------------------------------------------------------------
# Text cleaning (applied per-story after reading)
# ---------------------------------------------------------------------------

def clean_story(text: str, entry: dict, boilerplate: BoilerplateNode,
                rules: list[dict]) -> tuple[DataChunk, int, int]:
    """Run full cleaning pipeline on one story's text.

    Returns (chunk, original_chars, final_chars).
    """
    original_chars = len(text)

    # Extract metadata
    meta = extract_metadata(text, entry)
    chunk_meta = {**entry, **meta}

    # HTML strip
    if _is_html(text):
        text = strip_html(text)

    # Email headers
    text = strip_email_headers(text)

    # Boilerplate
    bp_chunk = DataChunk(text=text, metadata=chunk_meta)
    bp_result = boilerplate.process({'text': [bp_chunk]}, {
        'use_defaults': True, 'patterns': '', 'scope_lines': 0,
    })
    text = bp_result['cleaned'][0].text

    # Regex rules (135 rules)
    text, _ = apply_rules(text, rules)

    # Normalize
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = unicodedata.normalize('NFC', text)

    # Reflow
    text = reflow_text(text, 80)

    final_chars = len(text)

    # Note: chunking happens after quality filter, not here.
    # clean_story returns the full cleaned text as one chunk.

    chunk = DataChunk(
        text=text,
        metadata=chunk_meta,
        history=['full_pipeline'],
    )
    return chunk, original_chars, final_chars


# ---------------------------------------------------------------------------
# Chapter-aware chunking
# ---------------------------------------------------------------------------

def chunk_story(text: str, max_tokens: int = MAX_CHUNK_TOKENS,
                overlap_tokens: int = OVERLAP_TOKENS,
                chapter_sep: str = CHAPTER_SEP) -> list[str]:
    """Split a story into training-ready chunks.

    Strategy:
      1. Split on chapter separators (---) into chapters
      2. Group consecutive chapters into chunks that fit max_tokens
      3. If a single chapter exceeds max_tokens, use sliding window

    Returns list of text chunks. Short stories (under max_tokens) are
    returned as-is in a single-element list.
    """
    est_tokens = len(text) // 4
    if est_tokens <= max_tokens:
        return [text]

    # Split into chapters
    chapters = text.split(chapter_sep)
    chapters = [c.strip() for c in chapters if c.strip()]

    if len(chapters) <= 1:
        # Single chapter but too long — sliding window
        return _sliding_window(text, max_tokens, overlap_tokens)

    # Group chapters into chunks
    chunks = []
    current_parts = []
    current_tokens = 0

    for chapter in chapters:
        ch_tokens = len(chapter) // 4

        if ch_tokens > max_tokens:
            # Flush current accumulator first
            if current_parts:
                chunks.append(chapter_sep.join(current_parts))
                current_parts = []
                current_tokens = 0
            # Sliding window on the oversized chapter
            chunks.extend(_sliding_window(chapter, max_tokens, overlap_tokens))
            continue

        # Would adding this chapter exceed the limit?
        if current_tokens + ch_tokens > max_tokens and current_parts:
            # Flush current chunk
            chunks.append(chapter_sep.join(current_parts))
            current_parts = []
            current_tokens = 0

        current_parts.append(chapter)
        current_tokens += ch_tokens

    # Flush remaining
    if current_parts:
        chunks.append(chapter_sep.join(current_parts))

    return chunks


def _sliding_window(text: str, max_tokens: int,
                    overlap_tokens: int) -> list[str]:
    """Split oversized text using a sliding window with overlap.

    Tries to break at paragraph boundaries for cleaner chunks.
    """
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    stride = max_chars - overlap_chars

    if stride <= 0:
        stride = max_chars // 2

    chunks = []
    pos = 0

    while pos < len(text):
        end = pos + max_chars

        if end >= len(text):
            # Last chunk — take everything remaining
            chunk = text[pos:]
            if chunk.strip():
                chunks.append(chunk.strip())
            break

        # Try to break at a paragraph boundary (double newline)
        # Search backwards from the end for a clean break point
        search_start = max(pos + stride, end - 2000)
        break_point = text.rfind('\n\n', search_start, end)

        if break_point > pos + stride // 2:
            end = break_point
        else:
            # No paragraph break found — break at sentence end
            for sep in ['. ', '.\n', '! ', '?\n']:
                bp = text.rfind(sep, search_start, end)
                if bp > pos + stride // 2:
                    end = bp + len(sep)
                    break

        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)

        pos = end - overlap_chars

    return chunks


# ---------------------------------------------------------------------------
# Vault health check (auto-remount on macOS if SMB dropped)
# ---------------------------------------------------------------------------

def ensure_vault(vault_dir: str, max_retries: int = 3) -> bool:
    """Verify the vault is accessible. On macOS, attempt to remount if dropped.

    Returns True if vault is accessible.
    """
    import platform
    import subprocess

    # Quick check — can we list the directory?
    try:
        os.listdir(vault_dir)
        return True
    except OSError:
        pass

    log(f'  Vault not accessible at {vault_dir}')

    if platform.system() != 'Darwin':
        log(f'  Not macOS — cannot auto-remount. Check mount manually.')
        return False

    # macOS: try to remount
    # Infer the SMB share from the vault path
    # The mount point is the deepest existing ancestor
    mount_point = vault_dir
    while mount_point and not os.path.ismount(mount_point):
        mount_point = os.path.dirname(mount_point)
        if mount_point == '/':
            break

    if not mount_point or mount_point == '/':
        # Can't find mount point — try common locations
        home = os.path.expanduser('~')
        for candidate in [
            os.path.join(home, 'mnt', 'vault'),
            '/mnt/vault',
        ]:
            if vault_dir.startswith(candidate):
                mount_point = candidate
                break

    if not mount_point or mount_point == '/':
        log(f'  Cannot determine mount point for {vault_dir}')
        return False

    # Read SMB credentials from environment or use defaults
    smb_user = os.environ.get('VAULT_SMB_USER', 'corr')
    smb_pass = os.environ.get('VAULT_SMB_PASS', '1922')
    smb_host = os.environ.get('VAULT_SMB_HOST', '192.168.1.123')
    smb_share = os.environ.get('VAULT_SMB_SHARE', 'vault')

    for attempt in range(max_retries):
        log(f'  Remount attempt {attempt + 1}/{max_retries}...')
        try:
            # Unmount stale mount
            subprocess.run(['umount', mount_point],
                           capture_output=True, timeout=10)
            import time as _time
            _time.sleep(1)

            # Ensure mount point exists
            os.makedirs(mount_point, exist_ok=True)

            # Mount
            smb_url = f'//{smb_user}:{smb_pass}@{smb_host}/{smb_share}'
            r = subprocess.run(
                ['mount_smbfs', smb_url, mount_point],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode == 0:
                # Verify
                try:
                    os.listdir(vault_dir)
                    log(f'  Vault remounted successfully')
                    return True
                except OSError:
                    log(f'  Mount succeeded but vault_dir still not accessible')
            else:
                log(f'  mount_smbfs failed: {r.stderr.strip()}')
        except subprocess.TimeoutExpired:
            log(f'  Mount timed out')
        except Exception as e:
            log(f'  Remount error: {e}')

    log(f'  Failed to remount vault after {max_retries} attempts')
    return False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def main():
    start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: Scan vault
    # -----------------------------------------------------------------------
    # Pre-flight: ensure vault is mounted
    if not ensure_vault(VAULT_DIR):
        log('FATAL: Vault is not accessible and could not be remounted.')
        sys.exit(1)

    log(f'Phase 1: Scanning vault at {VAULT_DIR}')
    all_entries = scan_vault(VAULT_DIR)
    scan_time = time.time() - start
    log(f'  Found {len(all_entries):,} story entries in {scan_time:.0f}s')

    if not all_entries:
        log('ERROR: No stories found. Check vault mount.')
        sys.exit(1)

    # Load rules
    rules = load_rule_library('nifty_archive')
    log(f'  Loaded {len(rules)} cleaning rules')

    # Initialize nodes
    boilerplate = BoilerplateNode()
    quality = QualityFilterNode()

    # -----------------------------------------------------------------------
    # Phase 2: Parallel read + process in batches
    # -----------------------------------------------------------------------
    log(f'\nPhase 2: Processing {len(all_entries):,} stories '
        f'(batches of {BATCH_SIZE}, {IO_WORKERS} I/O threads)')

    # Temp file in /tmp to avoid Syncthing interference
    import hashlib
    import tempfile
    temp_jsonl = os.path.join(tempfile.gettempdir(), f'orracle_pipeline_{os.getpid()}.jsonl')
    seen_hashes = set()  # Only store 32-byte hashes, not full text

    total_read = 0
    total_cleaned = 0
    total_passed = 0
    total_chunks_written = 0
    total_rejected = 0
    total_skipped = 0
    total_read_errors = 0
    total_deduped = 0
    total_original_chars = 0
    total_final_chars = 0
    total_stories_chunked = 0
    category_counts = {}
    consecutive_empty_batches = 0

    n_batches = (len(all_entries) + BATCH_SIZE - 1) // BATCH_SIZE

    with open(temp_jsonl, 'w', encoding='utf-8') as out_f:
        for batch_idx in range(n_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(all_entries))
            batch_entries = all_entries[batch_start:batch_end]
            elapsed = time.time() - start

            pct = round((batch_idx / n_batches) * 100)
            rate = total_read / max(1, elapsed - scan_time) if elapsed > scan_time else 0
            eta_stories = len(all_entries) - batch_start
            eta_min = (eta_stories / max(1, rate)) / 60 if rate > 0 else 0
            log(f'  [{pct:3d}%] Batch {batch_idx+1}/{n_batches} '
                f'({batch_start:,}-{batch_end:,}, {elapsed:.0f}s, '
                f'{total_passed:,} passed, {rate:.1f} stories/s'
                f'{f", ~{eta_min:.0f}m left" if rate > 0 else ""})')

            # Parallel read from vault
            read_results = []
            with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
                futures = {pool.submit(read_entry, e): e for e in batch_entries}
                for future in as_completed(futures):
                    try:
                        entry, text, error = future.result()
                        read_results.append((entry, text, error))
                    except Exception as e:
                        total_read_errors += 1

            # Check for mass I/O failures (NAS dropped)
            batch_io_errors = sum(1 for _, _, err in read_results
                                  if err and err not in ('skip', 'empty_dir'))
            batch_reads = sum(1 for _, text, _ in read_results if text is not None)

            if batch_reads == 0 and batch_io_errors > len(batch_entries) * 0.5:
                consecutive_empty_batches += 1
                sample_errors = [err for _, _, err in read_results
                                 if err and err not in ('skip', 'empty_dir')][:3]
                log(f'    WARNING: Batch had 0 reads, {batch_io_errors} I/O errors')
                for e in sample_errors:
                    log(f'      Error: {e}')

                if consecutive_empty_batches >= 3:
                    log(f'    PAUSING: {consecutive_empty_batches} consecutive empty batches — NAS may be down')
                    # Try to remount before waiting
                    if ensure_vault(VAULT_DIR):
                        log(f'    Vault remounted — resuming')
                        consecutive_empty_batches = 0
                        continue
                    log(f'    Waiting 60s before retrying...')
                    time.sleep(60)
                    test_path = batch_entries[0]['path']
                    try:
                        with open(test_path, 'rb') as f:
                            f.read(1)
                        log(f'    NAS is back — resuming')
                        consecutive_empty_batches = 0
                    except OSError:
                        log(f'    NAS still down — continuing (will retry later if it recovers)')
            else:
                consecutive_empty_batches = 0

            # Process each story (single-threaded CPU work)
            processed = []
            for entry, text, error in read_results:
                if text is None:
                    if error in ('skip', 'empty_dir'):
                        total_skipped += 1
                    else:
                        total_read_errors += 1
                    continue

                total_read += 1
                cat = entry.get('category', 'unknown')
                category_counts[cat] = category_counts.get(cat, 0) + 1

                chunk, orig_chars, final_chars = clean_story(
                    text, entry, boilerplate, rules)
                total_original_chars += orig_chars
                total_final_chars += final_chars
                total_cleaned += 1
                processed.append(chunk)

            # Quality filter
            if processed:
                qf_result = quality.process({'text': processed}, {
                    'min_chars': MIN_CHARS,
                    'max_chars': 2_000_000,
                    'min_words': 50,
                    'max_non_ascii': 0.20,
                    'min_sentences': 3,
                    'max_avg_word_len': 15.0,
                })
                total_rejected += len(qf_result['rejected'])

                # Chunk + dedup + write to disk (no RAM accumulation)
                for story_chunk in qf_result['passed']:
                    # Deduplicate the full story first
                    text_hash = hashlib.sha256(story_chunk.text.encode()).digest()
                    if text_hash in seen_hashes:
                        total_deduped += 1
                        continue
                    seen_hashes.add(text_hash)
                    total_passed += 1

                    # Split into training-sized chunks
                    text_chunks = chunk_story(story_chunk.text)
                    if len(text_chunks) > 1:
                        total_stories_chunked += 1

                    for tc in text_chunks:
                        if len(tc.strip()) < MIN_CHARS:
                            continue  # Skip tiny trailing chunks
                        record = json.dumps({'text': tc}, ensure_ascii=False)
                        out_f.write(record + '\n')
                        total_chunks_written += 1

            # Free batch memory
            del read_results, processed

    # -----------------------------------------------------------------------
    # Phase 3: Shuffle + train/val split from temp file
    # -----------------------------------------------------------------------
    log(f'\nPhase 3: Split + export')
    log(f'  {total_passed:,} unique stories, {total_deduped:,} duplicates removed')

    # Read line offsets (not full text) for shuffling
    log(f'  Building line index...')
    line_offsets = []
    with open(temp_jsonl, 'rb') as f:
        offset = 0
        for line in f:
            line_offsets.append(offset)
            offset += len(line)

    # Shuffle indices
    rng = random.Random(SEED)
    indices = list(range(len(line_offsets)))
    rng.shuffle(indices)

    n_val = max(1, int(len(indices) * VAL_RATIO))
    val_indices = set(indices[:n_val])

    # Write train/val by reading lines from temp file
    train_path = os.path.join(OUTPUT_DIR, 'train.jsonl')
    val_path = os.path.join(OUTPUT_DIR, 'val.jsonl')

    train_tokens = 0
    val_tokens = 0
    train_count = 0
    val_count = 0

    log(f'  Writing train/val split...')
    with open(temp_jsonl, 'r', encoding='utf-8') as src, \
         open(train_path, 'w', encoding='utf-8') as train_f, \
         open(val_path, 'w', encoding='utf-8') as val_f:
        for i, line in enumerate(src):
            line = line.rstrip('\n')
            if not line:
                continue
            text_len = len(line)  # approximate
            if i in val_indices:
                val_f.write(line + '\n')
                val_tokens += text_len // 4
                val_count += 1
            else:
                train_f.write(line + '\n')
                train_tokens += text_len // 4
                train_count += 1

    # Clean up temp
    os.remove(temp_jsonl)

    elapsed = time.time() - start
    train_size = os.path.getsize(train_path)
    val_size = os.path.getsize(val_path)

    log(f'\n{"="*60}')
    log(f'PIPELINE COMPLETE')
    log(f'{"="*60}')
    log(f'Time:           {elapsed:.0f}s ({elapsed/60:.1f}m)')
    log(f'Stories scanned: {len(all_entries):,}')
    log(f'Stories read:    {total_read:,} (skipped: {total_skipped:,}, errors: {total_read_errors:,})')
    log(f'After cleaning:  {total_cleaned:,}')
    log(f'Quality passed:  {total_passed:,} (rejected {total_rejected:,})')
    log(f'Duplicates:      {total_deduped:,}')
    log(f'Chunking:        {total_stories_chunked:,} stories split into {total_chunks_written:,} chunks (max ~{MAX_CHUNK_TOKENS} tokens)')
    log(f'Train set:       {train_count:,} chunks ({train_size/(1024**2):.1f} MB, ~{train_tokens:,} tokens)')
    log(f'Val set:         {val_count:,} stories ({val_size/(1024**2):.1f} MB, ~{val_tokens:,} tokens)')
    log(f'Char reduction:  {total_original_chars:,} -> {total_final_chars:,} '
        f'({round((1-total_final_chars/max(1,total_original_chars))*100,1)}%)')
    log(f'Output:          {OUTPUT_DIR}/')
    log(f'Hashes in mem:   {len(seen_hashes):,} ({len(seen_hashes) * 32 / 1024:.0f} KB)')
    log(f'\nCategories:')
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        log(f'  {cat:20} {count:>6}')


if __name__ == '__main__':
    main()
