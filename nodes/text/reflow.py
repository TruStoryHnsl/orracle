"""Reassemble hard-wrapped text into paragraphs."""

from __future__ import annotations

import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType


def reflow_text(text: str, max_line_len: int = 80) -> str:
    """Unwrap hard-wrapped text while preserving paragraph boundaries.

    Lines shorter than max_line_len that don't end with sentence-ending
    punctuation are joined to the next line. Blank lines (paragraph
    boundaries) are preserved.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    result = []

    for para in paragraphs:
        lines = para.split('\n')
        if len(lines) <= 1:
            result.append(para)
            continue

        # Check if this looks like hard-wrapped prose
        avg_len = sum(len(l.rstrip()) for l in lines if l.strip()) / max(1, sum(1 for l in lines if l.strip()))
        if avg_len < 40:
            # Short lines — likely formatted text (poetry, code), don't reflow
            result.append(para)
            continue

        reflowed = []
        current = ''
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                if current:
                    reflowed.append(current)
                    current = ''
                reflowed.append('')
                continue

            if not current:
                current = stripped
            elif (len(current) > max_line_len * 0.5 and
                  not current.endswith(('.', '!', '?', ':', '"', "'", '\u201d')) and
                  not stripped[0].isupper()):
                # Continue the same line — likely hard-wrapped
                current = current + ' ' + stripped.lstrip()
            else:
                reflowed.append(current)
                current = stripped

        if current:
            reflowed.append(current)
        result.append('\n'.join(reflowed))

    return '\n\n'.join(result)


@NodeRegistry.register
class ReflowNode(BaseNode):
    node_type = 'reflow'
    label = 'Fix Line Breaks'
    category = 'text'
    description = 'Rejoin text that was broken into short lines by email or HTML formatting'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text chunks')]
    outputs = [
        Port('cleaned', PortType.TEXT_BATCH, 'Text with natural paragraph flow'),
        Port('metrics', PortType.METRICS, 'Reflow stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'max_line_len': {
                'type': 'integer',
                'title': 'Max line length',
                'description': 'Expected hard-wrap column width',
                'default': 80,
                'minimum': 40,
                'maximum': 200,
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        max_len = config.get('max_line_len', 80)
        out = []
        modified = 0

        for chunk in chunks:
            cleaned = reflow_text(chunk.text, max_len)
            if cleaned != chunk.text:
                modified += 1
            out.append(chunk.with_text(cleaned, 'reflow'))

        return {
            'cleaned': out,
            'out': out,
            'metrics': {'total': len(chunks), 'modified': modified},
        }
