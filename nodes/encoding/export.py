"""Export to JSONL — MLX, HF, and raw formats."""

from __future__ import annotations

import json
import os
from pathlib import Path

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType


def format_mlx(chunk: DataChunk) -> dict:
    """MLX-LM training format: {"text": "..."}"""
    return {'text': chunk.text}


def format_hf(chunk: DataChunk) -> dict:
    """HuggingFace datasets format with metadata."""
    return {
        'text': chunk.text,
        'source': chunk.metadata.get('source_file', chunk.metadata.get('name', '')),
        'category': chunk.metadata.get('category', ''),
    }


def format_raw(chunk: DataChunk) -> dict:
    """Full export with all metadata."""
    return {
        'text': chunk.text,
        'metadata': chunk.metadata,
        'history': chunk.history,
    }


_FORMATTERS = {
    'mlx': format_mlx,
    'huggingface': format_hf,
    'raw': format_raw,
}


@NodeRegistry.register
class ExportJsonlNode(BaseNode):
    node_type = 'export_jsonl'
    label = 'Export JSONL'
    category = 'encoding'
    description = 'Export text chunks to JSONL file (MLX, HuggingFace, or raw format)'
    inputs = [Port('in', PortType.TEXT_BATCH, 'Text chunks to export')]
    outputs = [
        Port('metrics', PortType.METRICS, 'Export stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'output_dir': {
                'type': 'string',
                'format': 'path',
                'title': 'Output directory',
                'default': './output',
            },
            'filename': {
                'type': 'string',
                'title': 'Filename',
                'default': 'train.jsonl',
            },
            'format': {
                'type': 'string',
                'title': 'Format',
                'enum': ['mlx', 'huggingface', 'raw'],
                'default': 'mlx',
            },
            'pretty': {
                'type': 'boolean',
                'title': 'Pretty print',
                'description': 'Format JSON with indentation (larger files)',
                'default': False,
            },
        },
        'required': ['output_dir'],
    }

    def process(self, inputs, config):
        chunks = inputs.get('in', [])
        output_dir = os.path.expanduser(config.get('output_dir', './output'))
        filename = config.get('filename', 'train.jsonl')
        fmt = config.get('format', 'mlx')
        pretty = config.get('pretty', False)

        formatter = _FORMATTERS.get(fmt, format_mlx)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        total_chars = 0
        total_tokens = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                record = formatter(chunk)
                if pretty:
                    line = json.dumps(record, ensure_ascii=False, indent=2)
                else:
                    line = json.dumps(record, ensure_ascii=False)
                f.write(line + '\n')
                total_chars += len(chunk.text)
                total_tokens += chunk.metadata.get('token_count', 0)

        file_size = os.path.getsize(output_path)

        return {
            'metrics': {
                'records': len(chunks),
                'output_path': output_path,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'total_chars': total_chars,
                'total_tokens': total_tokens,
                'format': fmt,
            },
        }

    def preview(self, inputs, config, n=3):
        """Preview: show formatted records without writing to disk."""
        chunks = inputs.get('in', [])[:n]
        fmt = config.get('format', 'mlx')
        formatter = _FORMATTERS.get(fmt, format_mlx)

        samples = []
        for chunk in chunks:
            record = formatter(chunk)
            samples.append(json.dumps(record, ensure_ascii=False, indent=2)[:500])

        return {
            'metrics': {
                'preview_records': len(samples),
                'samples': samples,
                'format': fmt,
            },
        }
