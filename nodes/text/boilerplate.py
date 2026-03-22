"""Remove configurable boilerplate patterns from text."""

from __future__ import annotations

import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType

# Default boilerplate patterns (Nifty Archive focused)
_DEFAULT_PATTERNS = [
    r'(?i)this\s+(?:story|work)\s+(?:is|was)\s+(?:posted|submitted|written)\s+(?:to|for|on)\s+(?:the\s+)?nifty',
    r'(?i)(?:donate|donation|support).*nifty|nifty.*(?:donate|donation|support)',
    r'(?i)nifty\.org|niftyarchive',
    r'(?i)^\s*(?:disclaimer|all rights reserved|reproduction.*prohibited)',
    r'(?i)(?:send|email|write)\s+(?:me|the author|us)\s+(?:at|to|your|feedback|comments)',
    r'(?i)^\s*(?:copyright|\(c\)|©)\s+\d{4}',
]


@NodeRegistry.register
class BoilerplateNode(BaseNode):
    node_type = 'boilerplate'
    label = 'Boilerplate Removal'
    category = 'text'
    description = 'Remove archive boilerplate: disclaimers, donation notices, submission headers'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text to clean')]
    outputs = [
        Port('cleaned', PortType.TEXT_BATCH, 'Text with boilerplate removed'),
        Port('metrics', PortType.METRICS, 'Removal stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'patterns': {
                'type': 'string',
                'title': 'Extra patterns',
                'description': 'Additional regex patterns (one per line)',
                'default': '',
            },
            'use_defaults': {
                'type': 'boolean',
                'title': 'Include default patterns',
                'description': 'Use built-in Nifty Archive boilerplate patterns',
                'default': True,
            },
            'scope_lines': {
                'type': 'integer',
                'title': 'Scope (lines)',
                'description': 'Only check first/last N lines (0 = entire text)',
                'default': 0,
                'minimum': 0,
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        use_defaults = config.get('use_defaults', True)
        extra = config.get('patterns', '').strip()
        scope = config.get('scope_lines', 0)

        patterns = []
        if use_defaults:
            patterns.extend(_DEFAULT_PATTERNS)
        if extra:
            for line in extra.split('\n'):
                line = line.strip()
                if line:
                    patterns.append(line)

        compiled = []
        for pat in patterns:
            try:
                compiled.append(re.compile(pat, re.MULTILINE))
            except re.error:
                continue

        out = []
        total_removed = 0

        for chunk in chunks:
            text = chunk.text
            lines = text.split('\n')
            removed = 0

            if scope > 0:
                # Only check first/last N lines
                check_indices = set(range(min(scope, len(lines))))
                check_indices |= set(range(max(0, len(lines) - scope), len(lines)))
            else:
                check_indices = set(range(len(lines)))

            keep = []
            for i, line in enumerate(lines):
                if i in check_indices:
                    matched = any(p.search(line) for p in compiled)
                    if matched:
                        removed += 1
                        continue
                keep.append(line)

            total_removed += removed
            cleaned = '\n'.join(keep)
            out.append(chunk.with_text(cleaned, 'boilerplate'))

        return {
            'cleaned': out,
            'out': out,
            'metrics': {
                'total': len(chunks),
                'lines_removed': total_removed,
                'patterns_active': len(compiled),
            },
        }
