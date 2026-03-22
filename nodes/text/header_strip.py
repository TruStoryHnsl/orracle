"""Strip RFC 2822 email headers from text tops."""

from __future__ import annotations

import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType

# RFC 2822 header pattern: "Key: Value" at start of text
_HEADER_RE = re.compile(
    r'^(?:(?:From|To|Cc|Bcc|Subject|Date|Reply-To|Message-ID|'
    r'Content-Type|MIME-Version|Return-Path|Received|X-[\w-]+)'
    r'\s*:.*\n?)+',
    re.MULTILINE | re.IGNORECASE,
)

# Mbox "From " line
_MBOX_RE = re.compile(r'^From\s+\S+.*\d{4}\s*\n', re.MULTILINE)


def strip_email_headers(text: str) -> str:
    """Remove email headers from the beginning of text."""
    # Only strip headers at the very start (within first 2000 chars)
    head = text[:2000]
    rest = text[2000:]

    head = _MBOX_RE.sub('', head)
    head = _HEADER_RE.sub('', head)

    result = head + rest
    return result.lstrip('\n')


@NodeRegistry.register
class HeaderStripNode(BaseNode):
    node_type = 'header_strip'
    label = 'Remove Email Headers'
    category = 'text'
    description = 'Remove email-style headers (From, Subject, Date) from the top of text files'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text that may start with email headers')]
    outputs = [
        Port('cleaned', PortType.TEXT_BATCH, 'Text with headers removed'),
        Port('metrics', PortType.METRICS, 'How many files had headers', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {},
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        out = []
        modified = 0

        for chunk in chunks:
            cleaned = strip_email_headers(chunk.text)
            if cleaned != chunk.text:
                modified += 1
            out.append(chunk.with_text(cleaned, 'header_strip'))

        return {
            'cleaned': out,
            'out': out,
            'metrics': {'total': len(chunks), 'modified': modified},
        }
