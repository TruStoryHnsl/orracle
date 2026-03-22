"""Pollution detection — 9+ togglable detector patterns for spotting remaining junk."""

from __future__ import annotations

import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType

DETECTORS = {
    'repeated_chars': {
        'name': 'Repeated character lines',
        'pattern': re.compile(r'^(.)\1{5,}\s*$', re.MULTILINE),
    },
    'url_density': {
        'name': 'URLs in text',
        'pattern': re.compile(r'https?://[^\s<>"\')\]]{4,}'),
    },
    'nifty_boilerplate': {
        'name': 'Nifty Archive boilerplate',
        'pattern': re.compile(
            r'(?i)nifty\.org|niftyarchive|donate.*nifty|nifty.*erotic',
        ),
    },
    'divider_lines': {
        'name': 'Divider/separator lines',
        'pattern': re.compile(r'^[-=~*_]{3,}\s*$', re.MULTILINE),
    },
    'nav_text': {
        'name': 'Navigation text',
        'pattern': re.compile(
            r'(?i)(?:next|previous|prev)\s+(?:chapter|part|page)',
        ),
    },
    'email_addresses': {
        'name': 'Email addresses',
        'pattern': re.compile(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    },
    'copyright_notices': {
        'name': 'Copyright notices',
        'pattern': re.compile(r'(?i)(?:copyright|\(c\)|©)\s+\d{4}'),
    },
    'non_printable': {
        'name': 'Non-printable characters',
        'pattern': re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]'),
    },
    'html_remnants': {
        'name': 'HTML tag remnants',
        'pattern': re.compile(r'</?[a-zA-Z][^>]*>'),
    },
}


def scan_text(text: str, enabled: set[str] | None = None) -> dict:
    """Scan text for pollution. Returns {detector_id: [matches]}."""
    results = {}
    for det_id, det in DETECTORS.items():
        if enabled is not None and det_id not in enabled:
            continue
        matches = det['pattern'].findall(text)
        if matches:
            results[det_id] = {
                'name': det['name'],
                'count': len(matches),
                'samples': [m[:100] if isinstance(m, str) else str(m)[:100]
                            for m in matches[:5]],
            }
    return results


@NodeRegistry.register
class PollutionScanNode(BaseNode):
    node_type = 'pollution_scan'
    label = 'Junk Detector'
    category = 'text'
    description = 'Find remaining unwanted content — URLs, boilerplate remnants, HTML fragments, control characters'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text chunks')]
    outputs = [
        Port('clean', PortType.TEXT_BATCH, 'Clean chunks (no detections)'),
        Port('flagged', PortType.TEXT_BATCH, 'Flagged chunks', required=False),
        Port('metrics', PortType.METRICS, 'Pollution stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'detectors': {
                'type': 'string',
                'title': 'Active detectors',
                'description': 'Comma-separated detector IDs (empty = all)',
                'default': '',
            },
            'threshold': {
                'type': 'integer',
                'title': 'Flag threshold',
                'description': 'Min total hits to flag a chunk as polluted',
                'default': 1,
                'minimum': 1,
            },
            'mode': {
                'type': 'string',
                'title': 'Mode',
                'enum': ['filter', 'report_only'],
                'default': 'filter',
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        det_filter = config.get('detectors', '').strip()
        threshold = config.get('threshold', 1)
        mode = config.get('mode', 'filter')

        enabled = None
        if det_filter:
            enabled = {d.strip() for d in det_filter.split(',') if d.strip()}

        clean = []
        flagged = []
        all_hits = {}

        for chunk in chunks:
            results = scan_text(chunk.text, enabled)
            total_hits = sum(r['count'] for r in results.values())

            if total_hits >= threshold:
                flagged.append(chunk)
                for det_id, info in results.items():
                    if det_id not in all_hits:
                        all_hits[det_id] = {'name': info['name'], 'count': 0}
                    all_hits[det_id]['count'] += info['count']
            else:
                clean.append(chunk)

        # In report_only mode, pass everything through
        if mode == 'report_only':
            clean = chunks
            flagged = []

        return {
            'clean': clean,
            'flagged': flagged,
            'metrics': {
                'total': len(chunks),
                'clean': len(clean),
                'flagged': len(flagged),
                'detectors': all_hits,
            },
        }
