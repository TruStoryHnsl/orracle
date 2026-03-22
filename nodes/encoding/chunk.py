"""Sliding window chunking with boundary-aware splitting."""

from __future__ import annotations

import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType


def chunk_text(text: str, chunk_size: int, overlap: int, boundary: str) -> list[str]:
    """Split text into overlapping chunks respecting boundaries."""
    if not text.strip():
        return []

    if boundary == 'paragraph':
        segments = re.split(r'\n\s*\n', text)
    elif boundary == 'sentence':
        segments = re.split(r'(?<=[.!?])\s+', text)
    else:
        # Character-level splitting
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    # Boundary-aware: accumulate segments until chunk_size
    chunks = []
    current = ''
    for seg in segments:
        sep = '\n\n' if boundary == 'paragraph' else ' '
        candidate = (current + sep + seg).strip() if current else seg.strip()
        if len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            current = seg.strip()
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks


@NodeRegistry.register
class ChunkNode(BaseNode):
    node_type = 'chunk'
    label = 'Chunk Text'
    category = 'encoding'
    description = 'Split text into chunks with configurable size, overlap, and boundary mode'
    inputs = [Port('in', PortType.TEXT_BATCH, 'Text chunks')]
    outputs = [
        Port('out', PortType.TEXT_BATCH, 'Chunked text'),
        Port('metrics', PortType.METRICS, 'Chunk stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'chunk_size': {
                'type': 'integer',
                'title': 'Chunk size (chars)',
                'default': 4096,
                'minimum': 100,
                'maximum': 100000,
            },
            'overlap': {
                'type': 'integer',
                'title': 'Overlap (chars)',
                'default': 0,
                'minimum': 0,
            },
            'boundary': {
                'type': 'string',
                'title': 'Split boundary',
                'enum': ['paragraph', 'sentence', 'character'],
                'default': 'paragraph',
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('in', [])
        size = config.get('chunk_size', 4096)
        overlap = config.get('overlap', 0)
        boundary = config.get('boundary', 'paragraph')

        out = []
        total_input = 0
        total_output = 0

        for i, chunk in enumerate(chunks):
            total_input += 1
            pieces = chunk_text(chunk.text, size, overlap, boundary)
            for j, piece in enumerate(pieces):
                total_output += 1
                out.append(DataChunk(
                    text=piece,
                    metadata={
                        **chunk.metadata,
                        'chunk_index': j,
                        'chunk_total': len(pieces),
                        'source_index': i,
                    },
                    history=[*chunk.history, 'chunk'],
                ))

        return {
            'out': out,
            'metrics': {
                'input_chunks': total_input,
                'output_chunks': total_output,
                'avg_chunks_per_input': round(total_output / max(1, total_input), 1),
            },
        }
