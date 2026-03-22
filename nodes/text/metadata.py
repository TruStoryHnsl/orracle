"""Extract structural metadata from raw text before cleaning destroys it.

Extracts: title, author, chapter number, series name, category,
chapter break count, language hints, and word count.
Must run BEFORE boilerplate/pattern cleaning nodes.
"""

from __future__ import annotations

import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType

# Email header pattern (to skip when finding title)
_HEADER_RE = re.compile(
    r'^(?:From|To|Cc|Subject|Date|Reply-To|Content-Type|MIME-Version|Return-Path)\s*:',
    re.IGNORECASE,
)

# Chapter number from filename: story-name-42.html → 42
_FILENAME_CHAPTER_RE = re.compile(r'-(\d+)\.(?:txt|html|htm)$')

# "by Author Name" pattern
_BY_AUTHOR_RE = re.compile(r'(?i)^\s*(?:by|written\s+by|author:\s*)\s+(.+)$')

# Chapter/part markers in text
_CHAPTER_MARKER_RE = re.compile(
    r'(?i)(?:^|\n)\s*(?:chapter|part|section|episode)\s+(\d+|[IVXLC]+|one|two|three|four|five|six|seven|eight|nine|ten)',
)

# Story separator markers
_SEPARATOR_RE = re.compile(r'^\s*(?:---|\*\*\*|===|~~~)\s*$', re.MULTILINE)

# Non-ASCII ratio for language hinting
def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def extract_metadata(text: str, file_metadata: dict) -> dict:
    """Extract structural metadata from raw text + file info.

    Returns a dict of extracted fields to merge into chunk metadata.
    """
    meta = {}

    # From file path metadata
    path = file_metadata.get('path', '')
    relative = file_metadata.get('relative', '')
    name = file_metadata.get('name', '')
    category = file_metadata.get('category', '')

    if category:
        meta['category'] = category

    # Series detection: if file is in a subdirectory, that's the series
    parts = relative.split('/') if relative else []
    if len(parts) >= 2:
        meta['series'] = parts[-2] if len(parts) > 1 else ''
        meta['series_slug'] = meta['series']
    else:
        meta['series'] = ''

    # Chapter number from filename
    m = _FILENAME_CHAPTER_RE.search(name)
    if m:
        meta['chapter_num'] = int(m.group(1))
        meta['is_chapter'] = True
    else:
        meta['is_chapter'] = False

    # Extract title and author from first lines of text
    lines = text.strip().split('\n')
    title = None
    author = None

    for line in lines[:30]:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip email headers
        if _HEADER_RE.match(stripped):
            continue
        # Skip very short lines (artifacts)
        if len(stripped) < 3:
            continue
        # Skip boilerplate-looking lines
        if re.match(r'(?i)^\s*(?:this\s+story|disclaimer|warning|copyright|donation|nifty)', stripped):
            continue

        # First substantial line = title candidate
        if title is None and len(stripped) < 150:
            title = stripped
            continue

        # Check for "by Author"
        by_match = _BY_AUTHOR_RE.match(stripped)
        if by_match and not author:
            author = by_match.group(1).strip()
            # Clean up author name
            author = re.sub(r'\s*\(.*\)\s*$', '', author)  # Remove parenthetical
            author = re.sub(r'\s*<.*>\s*$', '', author)    # Remove email
            if len(author) > 60:
                author = None  # Probably not actually an author name
            break

    if title:
        # Clean chapter info from title for a "clean" title
        clean_title = re.sub(r'\s*(?:chapter|part|ch\.?)\s*\d+\s*$', '', title, flags=re.IGNORECASE).strip()
        clean_title = re.sub(r'\s*#?\d+\s*$', '', clean_title).strip()
        meta['title'] = title
        meta['title_clean'] = clean_title if clean_title else title
    if author:
        meta['author'] = author

    # Chapter breaks within text
    chapter_markers = _CHAPTER_MARKER_RE.findall(text)
    meta['chapter_breaks'] = len(chapter_markers)

    # Separator count
    meta['separator_count'] = len(_SEPARATOR_RE.findall(text))

    # Basic stats (before cleaning)
    meta['raw_chars'] = len(text)
    meta['raw_words'] = len(text.split())
    meta['raw_lines'] = text.count('\n') + 1

    # Language hint
    if category == 'non-english':
        meta['language'] = 'non-english'
    elif _non_ascii_ratio(text[:2000]) > 0.05:
        meta['language'] = 'non-english'
    else:
        meta['language'] = 'english'

    return meta


@NodeRegistry.register
class MetadataExtractNode(BaseNode):
    node_type = 'metadata_extract'
    label = 'Extract Metadata'
    category = 'text'
    description = 'Extract title, author, chapter info, and series data from raw text — run this BEFORE cleaning nodes'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Raw text chunks (before cleaning)')]
    outputs = [
        Port('enriched', PortType.TEXT_BATCH, 'Same text with metadata added'),
        Port('metrics', PortType.METRICS, 'Extraction stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'extract_title': {
                'type': 'boolean',
                'title': 'Extract title',
                'description': 'Find the story title from first lines',
                'default': True,
            },
            'extract_author': {
                'type': 'boolean',
                'title': 'Extract author',
                'description': 'Find author name from "by Author" lines',
                'default': True,
            },
            'detect_series': {
                'type': 'boolean',
                'title': 'Detect series',
                'description': 'Identify series and chapter from file path',
                'default': True,
            },
            'detect_language': {
                'type': 'boolean',
                'title': 'Detect language',
                'description': 'Flag non-English text',
                'default': True,
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))

        out = []
        titles_found = 0
        authors_found = 0
        series_found = 0
        chapters_found = 0

        for chunk in chunks:
            extracted = extract_metadata(chunk.text, chunk.metadata)

            # Apply config toggles
            if not config.get('extract_title', True):
                extracted.pop('title', None)
                extracted.pop('title_clean', None)
            if not config.get('extract_author', True):
                extracted.pop('author', None)
            if not config.get('detect_series', True):
                extracted.pop('series', None)
                extracted.pop('series_slug', None)
                extracted.pop('is_chapter', None)
                extracted.pop('chapter_num', None)
            if not config.get('detect_language', True):
                extracted.pop('language', None)

            # Merge into existing metadata
            new_meta = {**chunk.metadata, **extracted}
            new_chunk = DataChunk(
                text=chunk.text,
                metadata=new_meta,
                history=[*chunk.history, 'metadata_extract'],
            )
            out.append(new_chunk)

            # Count stats
            if extracted.get('title'):
                titles_found += 1
            if extracted.get('author'):
                authors_found += 1
            if extracted.get('series'):
                series_found += 1
            if extracted.get('is_chapter'):
                chapters_found += 1

        return {
            'enriched': out,
            'out': out,
            'metrics': {
                'total': len(chunks),
                'titles_found': titles_found,
                'authors_found': authors_found,
                'series_found': series_found,
                'chapters_detected': chapters_found,
            },
        }
