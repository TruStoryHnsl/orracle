"""Strip HTML tags, convert block elements to line breaks."""

from __future__ import annotations

import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType

# Block-level elements that produce line breaks
_BLOCK_TAGS = re.compile(
    r'<\s*/?\s*(?:p|div|br|h[1-6]|li|ul|ol|table|tr|td|th|'
    r'blockquote|pre|hr|section|article|header|footer|nav|aside)\b[^>]*>',
    re.IGNORECASE,
)
_ALL_TAGS = re.compile(r'<[^>]+>')
_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&apos;': "'", '&#39;': "'", '&nbsp;': ' ', '&#160;': ' ',
    '&mdash;': '\u2014', '&ndash;': '\u2013', '&hellip;': '\u2026',
    '&ldquo;': '\u201c', '&rdquo;': '\u201d',
    '&lsquo;': '\u2018', '&rsquo;': '\u2019',
}
_ENTITY_RE = re.compile(r'&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);')


def _decode_entity(m):
    entity = m.group(0)
    if entity in _ENTITIES:
        return _ENTITIES[entity]
    if entity.startswith('&#x'):
        try:
            return chr(int(entity[3:-1], 16))
        except (ValueError, OverflowError):
            return ''
    if entity.startswith('&#'):
        try:
            return chr(int(entity[2:-1]))
        except (ValueError, OverflowError):
            return ''
    return ''


def strip_html(text: str) -> str:
    """Remove HTML tags from text, preserving block structure."""
    # Remove script/style blocks entirely
    text = re.sub(r'<\s*(?:script|style)\b[^>]*>.*?</\s*(?:script|style)\s*>',
                  '', text, flags=re.IGNORECASE | re.DOTALL)
    # Block tags → newlines
    text = _BLOCK_TAGS.sub('\n', text)
    # Strip remaining tags
    text = _ALL_TAGS.sub('', text)
    # Decode entities
    text = _ENTITY_RE.sub(_decode_entity, text)
    return text


def _is_html(text: str) -> bool:
    """Quick check if text contains significant HTML."""
    tag_count = len(_ALL_TAGS.findall(text[:2000]))
    return tag_count > 3


@NodeRegistry.register
class HtmlStripNode(BaseNode):
    node_type = 'html_strip'
    label = 'Remove HTML'
    category = 'text'
    description = 'Strip HTML tags and convert web formatting to plain text'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text that may contain HTML')]
    outputs = [
        Port('cleaned', PortType.TEXT_BATCH, 'Plain text output'),
        Port('metrics', PortType.METRICS, 'How many files had HTML', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'skip_plain': {
                'type': 'boolean',
                'title': 'Skip plain text',
                'description': 'Only process chunks that contain HTML',
                'default': True,
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        skip_plain = config.get('skip_plain', True)
        out = []
        stripped_count = 0

        for chunk in chunks:
            if skip_plain and not _is_html(chunk.text):
                out.append(chunk)
            else:
                cleaned = strip_html(chunk.text)
                out.append(chunk.with_text(cleaned, 'html_strip'))
                stripped_count += 1

        return {
            'cleaned': out,
            'out': out,
            'metrics': {
                'total': len(chunks),
                'html_detected': stripped_count,
                'plain_passthrough': len(chunks) - stripped_count,
            },
        }
