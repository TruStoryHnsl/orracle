"""Quality filtering — char count, word count, non-ASCII ratio, sentence stats."""

from __future__ import annotations

import re

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType

_SENTENCE_RE = re.compile(r'[.!?]+\s')


@NodeRegistry.register
class QualityFilterNode(BaseNode):
    node_type = 'quality_filter'
    label = 'Quality Filter'
    category = 'text'
    description = 'Filter chunks by length, word count, non-ASCII ratio, and sentence structure'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text chunks')]
    outputs = [
        Port('passed', PortType.TEXT_BATCH, 'Chunks passing quality checks'),
        Port('rejected', PortType.TEXT_BATCH, 'Rejected chunks', required=False),
        Port('metrics', PortType.METRICS, 'Quality stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'min_chars': {
                'type': 'integer',
                'title': 'Min characters',
                'default': 500,
                'minimum': 0,
            },
            'max_chars': {
                'type': 'integer',
                'title': 'Max characters',
                'default': 2000000,
                'minimum': 100,
            },
            'min_words': {
                'type': 'integer',
                'title': 'Min words',
                'default': 50,
                'minimum': 0,
            },
            'max_non_ascii': {
                'type': 'number',
                'title': 'Max non-ASCII ratio',
                'description': 'Maximum fraction of non-ASCII characters',
                'default': 0.20,
                'minimum': 0.0,
                'maximum': 1.0,
            },
            'min_sentences': {
                'type': 'integer',
                'title': 'Min sentences',
                'default': 3,
                'minimum': 0,
            },
            'max_avg_word_len': {
                'type': 'number',
                'title': 'Max avg word length',
                'description': 'Flag garbled text with unreasonably long average words',
                'default': 15.0,
                'minimum': 5.0,
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        min_chars = config.get('min_chars', 500)
        max_chars = config.get('max_chars', 2_000_000)
        min_words = config.get('min_words', 50)
        max_non_ascii = config.get('max_non_ascii', 0.20)
        min_sentences = config.get('min_sentences', 3)
        max_avg_word = config.get('max_avg_word_len', 15.0)

        passed = []
        rejected = []
        reasons = {}

        for chunk in chunks:
            text = chunk.text
            reject_reason = None

            char_count = len(text)
            if char_count < min_chars:
                reject_reason = 'too_short'
            elif char_count > max_chars:
                reject_reason = 'too_long'

            if not reject_reason:
                words = text.split()
                word_count = len(words)
                if word_count < min_words:
                    reject_reason = 'too_few_words'
                elif word_count > 0:
                    avg_word = sum(len(w) for w in words) / word_count
                    if avg_word > max_avg_word:
                        reject_reason = 'garbled_text'

            if not reject_reason:
                non_ascii = sum(1 for c in text if ord(c) > 127)
                if char_count > 0 and non_ascii / char_count > max_non_ascii:
                    reject_reason = 'high_non_ascii'

            if not reject_reason:
                sentence_count = len(_SENTENCE_RE.findall(text))
                if sentence_count < min_sentences:
                    reject_reason = 'too_few_sentences'

            if reject_reason:
                rejected.append(chunk)
                reasons[reject_reason] = reasons.get(reject_reason, 0) + 1
            else:
                passed.append(chunk)

        return {
            'passed': passed,
            'rejected': rejected,
            'metrics': {
                'total': len(chunks),
                'passed': len(passed),
                'rejected': len(rejected),
                'pass_rate': round(len(passed) / max(1, len(chunks)) * 100, 1),
                'rejection_reasons': reasons,
            },
        }
