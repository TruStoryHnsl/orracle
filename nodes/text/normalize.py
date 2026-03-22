"""Unicode normalization, encoding fixes, control character removal."""

from __future__ import annotations

import re
import unicodedata

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType

# Control characters to strip (keep \n, \t, \r)
_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Common encoding mojibake fixes
_MOJIBAKE = {
    '\u00e2\u0080\u0099': '\u2019',  # right single quote
    '\u00e2\u0080\u009c': '\u201c',  # left double quote
    '\u00e2\u0080\u009d': '\u201d',  # right double quote
    '\u00e2\u0080\u0093': '\u2013',  # en dash
    '\u00e2\u0080\u0094': '\u2014',  # em dash
    '\u00e2\u0080\u00a6': '\u2026',  # ellipsis
    '\u00c2\u00a0': ' ',             # non-breaking space
}

# Smart quotes → straight quotes (optional)
_SMART_QUOTES = {
    '\u201c': '"', '\u201d': '"',  # double
    '\u2018': "'", '\u2019': "'",  # single
    '\u201e': '"', '\u201f': '"',  # double low/high
    '\u2039': "'", '\u203a': "'",  # single guillemets
    '\u00ab': '"', '\u00bb': '"',  # double guillemets
}


@NodeRegistry.register
class NormalizeNode(BaseNode):
    node_type = 'normalize'
    label = 'Fix Encoding'
    category = 'text'
    description = 'Fix garbled characters, encoding errors, and invisible control characters'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text chunks')]
    outputs = [
        Port('cleaned', PortType.TEXT_BATCH, 'Normalized text chunks'),
        Port('metrics', PortType.METRICS, 'Normalization stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'form': {
                'type': 'string',
                'title': 'Unicode form',
                'enum': ['NFC', 'NFKC', 'NFD', 'NFKD', 'none'],
                'default': 'NFC',
            },
            'fix_mojibake': {
                'type': 'boolean',
                'title': 'Fix mojibake',
                'description': 'Fix common double-encoded UTF-8 artifacts',
                'default': True,
            },
            'straighten_quotes': {
                'type': 'boolean',
                'title': 'Straighten quotes',
                'description': 'Convert smart/curly quotes to straight ASCII',
                'default': False,
            },
            'strip_control': {
                'type': 'boolean',
                'title': 'Strip control chars',
                'description': 'Remove non-printable control characters',
                'default': True,
            },
            'strip_replacement': {
                'type': 'boolean',
                'title': 'Strip replacement chars',
                'description': 'Remove Unicode replacement character (U+FFFD)',
                'default': True,
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        form = config.get('form', 'NFC')
        fix_mojibake = config.get('fix_mojibake', True)
        straighten = config.get('straighten_quotes', False)
        strip_ctrl = config.get('strip_control', True)
        strip_repl = config.get('strip_replacement', True)

        out = []
        fixes = 0

        for chunk in chunks:
            text = chunk.text
            changed = False

            if fix_mojibake:
                for bad, good in _MOJIBAKE.items():
                    if bad in text:
                        text = text.replace(bad, good)
                        changed = True

            if straighten:
                for smart, straight in _SMART_QUOTES.items():
                    if smart in text:
                        text = text.replace(smart, straight)
                        changed = True

            if strip_ctrl:
                new = _CONTROL_RE.sub('', text)
                if new != text:
                    text = new
                    changed = True

            if strip_repl and '\ufffd' in text:
                text = text.replace('\ufffd', '')
                changed = True

            if form != 'none':
                normalized = unicodedata.normalize(form, text)
                if normalized != text:
                    text = normalized
                    changed = True

            if changed:
                fixes += 1
            out.append(chunk.with_text(text, 'normalize'))

        return {
            'cleaned': out,
            'out': out,
            'metrics': {'total': len(chunks), 'modified': fixes},
        }
