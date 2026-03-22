"""Source nodes — file and directory input."""

from __future__ import annotations

import os
import random
from pathlib import Path

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType


def _is_text_file(fp: Path) -> bool:
    """Quick check: is this file likely readable text (not binary)?"""
    try:
        with open(fp, 'rb') as f:
            sample = f.read(512)
        return b'\x00' not in sample and len(sample) > 0
    except OSError:
        return False


@NodeRegistry.register
class DirectorySource(BaseNode):
    node_type = 'dir_source'
    label = 'Directory Source'
    category = 'source'
    description = 'Scan a directory for story files and multi-chapter folders'
    inputs = []
    outputs = [
        Port('files', PortType.FILES, 'List of discovered story paths'),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'directory': {
                'type': 'string',
                'format': 'path',
                'title': 'Directory',
                'description': 'Root directory to scan',
            },
            'extensions': {
                'type': 'string',
                'title': 'Extensions',
                'description': 'Comma-separated (empty = all text files)',
                'default': '',
            },
            'recursive': {
                'type': 'boolean',
                'title': 'Recursive',
                'description': 'Scan subdirectories for stories',
                'default': True,
            },
            'shuffle': {
                'type': 'boolean',
                'title': 'Random sample',
                'description': 'Randomly sample files instead of sorted order',
                'default': False,
            },
            'include_dirs': {
                'type': 'boolean',
                'title': 'Include directories',
                'description': 'Treat subdirectories as multi-chapter stories',
                'default': True,
            },
        },
        'required': ['directory'],
    }

    def _scan(self, config) -> list[dict]:
        """Scan directory and return list of story entries."""
        directory = os.path.expanduser(config.get('directory', ''))
        ext_str = config.get('extensions', '').strip()
        extensions = {e.strip().lower() for e in ext_str.split(',') if e.strip()} if ext_str else None
        recursive = config.get('recursive', True)
        include_dirs = config.get('include_dirs', True)

        root = Path(directory)
        if not root.exists():
            return []

        entries = []

        if recursive:
            # Walk category subdirectories, enumerate story entries
            for cat_dir in sorted(root.iterdir()):
                if not cat_dir.is_dir() or cat_dir.name.startswith('.'):
                    continue
                category = cat_dir.name
                for item in sorted(cat_dir.iterdir()):
                    if item.name.startswith('.'):
                        continue
                    entry = self._make_entry(item, root, category, extensions, include_dirs)
                    if entry:
                        entries.append(entry)
        else:
            for item in sorted(root.iterdir()):
                if item.name.startswith('.'):
                    continue
                entry = self._make_entry(item, root, '', extensions, include_dirs)
                if entry:
                    entries.append(entry)

        return entries

    def _make_entry(self, item: Path, root: Path, category: str,
                    extensions: set | None, include_dirs: bool) -> dict | None:
        """Create a file entry dict from a path, or None if it should be skipped."""
        if item.is_file():
            # Extension filter
            if extensions:
                if item.suffix.lower() not in extensions:
                    return None
            else:
                # No extension filter — accept text files (including extensionless)
                if item.suffix.lower() in ('.jpg', '.png', '.gif', '.pdf',
                                           '.css', '.js', '.aria2__temp',
                                           '.ds_store'):
                    return None
                if not _is_text_file(item):
                    return None

            return {
                'path': str(item),
                'name': item.name,
                'size': item.stat().st_size,
                'relative': str(item.relative_to(root)),
                'category': category,
                'type': 'file',
            }

        elif item.is_dir() and include_dirs:
            # Multi-chapter story directory
            children = [c for c in item.iterdir()
                        if c.is_file() and not c.name.startswith('.')]
            if not children:
                return None
            total_size = sum(c.stat().st_size for c in children)
            return {
                'path': str(item),
                'name': item.name,
                'size': total_size,
                'relative': str(item.relative_to(root)),
                'category': category,
                'type': 'directory',
                'chapter_count': len(children),
            }

        return None

    def process(self, inputs, config):
        entries = self._scan(config)
        if config.get('shuffle', False):
            random.shuffle(entries)
        return {'files': entries}

    def preview(self, inputs, config, n=10):
        if config.get('shuffle', False):
            # Reservoir sampling
            all_entries = self._scan(config)
            if len(all_entries) <= n:
                return {'files': all_entries}
            return {'files': random.sample(all_entries, n)}
        else:
            entries = self._scan(config)
            return {'files': entries[:n]}


@NodeRegistry.register
class FileReader(BaseNode):
    node_type = 'file_reader'
    label = 'File Reader'
    category = 'source'
    description = 'Read file contents into text chunks — handles single files and multi-chapter directories'
    inputs = [
        Port('files', PortType.FILES, 'File paths to read'),
    ]
    outputs = [
        Port('batch', PortType.TEXT_BATCH, 'Text content as DataChunks'),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'encoding': {
                'type': 'string',
                'title': 'Encoding',
                'default': 'utf-8',
                'enum': ['utf-8', 'latin-1', 'ascii', 'cp1252'],
            },
            'pdf_extract': {
                'type': 'boolean',
                'title': 'Extract PDF text',
                'description': 'Use pdfplumber for .pdf files',
                'default': True,
            },
            'chapter_separator': {
                'type': 'string',
                'title': 'Chapter separator',
                'description': 'Text inserted between chapters in multi-file stories',
                'default': '\\n\\n---\\n\\n',
            },
        },
    }

    def process(self, inputs, config):
        files = inputs.get('files', [])
        encoding = config.get('encoding', 'utf-8')
        pdf_extract = config.get('pdf_extract', True)
        sep = config.get('chapter_separator', '\n\n---\n\n').replace('\\n', '\n')

        chunks = []
        for finfo in files:
            path = finfo['path'] if isinstance(finfo, dict) else finfo
            fp = Path(path)

            if fp.is_dir():
                # Multi-chapter: read all files in dir, join
                texts = []
                for child in sorted(fp.iterdir()):
                    if child.is_file() and not child.name.startswith('.'):
                        t = self._read_one(child, encoding, pdf_extract)
                        if t is not None:
                            texts.append(t)
                if texts:
                    combined = sep.join(texts)
                    meta = finfo if isinstance(finfo, dict) else {'path': path}
                    chunks.append(DataChunk(
                        text=combined,
                        metadata={**meta, 'chapters': len(texts)},
                    ))
            elif fp.is_file():
                text = self._read_one(fp, encoding, pdf_extract)
                if text is not None:
                    meta = finfo if isinstance(finfo, dict) else {'path': path}
                    chunks.append(DataChunk(
                        text=text,
                        metadata={**meta, 'chapters': 1},
                    ))

        return {'batch': chunks}

    def _read_one(self, fp: Path, encoding: str, pdf_extract: bool) -> str | None:
        if fp.suffix.lower() == '.pdf' and pdf_extract:
            try:
                import pdfplumber
                pages = []
                with pdfplumber.open(str(fp)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            pages.append(t)
                return '\n\n'.join(pages) if pages else None
            except Exception:
                return None
        else:
            try:
                with open(fp, 'rb') as f:
                    sample = f.read(4096)
                if b'\x00' in sample:
                    return None
                return fp.read_text(encoding=encoding, errors='replace')
            except Exception:
                return None
