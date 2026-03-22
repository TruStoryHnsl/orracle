"""Rule engine — loads YAML rule libraries and applies them to text.

Supported actions:
  strip_line      — Remove any line matching the pattern
  strip_match     — Remove the matched text, keep surrounding content
  replace         — Replace matched text with a replacement string
  strip_paragraph — Join paragraph lines into one string, remove entire
                    paragraph if it matches (solves broken-line boilerplate)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
RULES_DIR = os.path.join(CONFIG_DIR, 'rules')


def load_rule_library(name: str) -> list[dict]:
    """Load rules from a YAML library file."""
    path = os.path.join(RULES_DIR, name)
    if not path.endswith('.yaml'):
        path += '.yaml'
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get('rules', [])


def list_rule_libraries() -> list[dict]:
    """List available rule library files."""
    if not os.path.isdir(RULES_DIR):
        return []
    libs = []
    for f in sorted(os.listdir(RULES_DIR)):
        if f.endswith('.yaml'):
            path = os.path.join(RULES_DIR, f)
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            rules = data.get('rules', [])
            libs.append({
                'name': f.replace('.yaml', ''),
                'filename': f,
                'rule_count': len(rules),
                'categories': sorted(set(r.get('category', '') for r in rules)),
            })
    return libs


def apply_rules(text: str, rules: list[dict]) -> tuple[str, int]:
    """Apply a list of rules to text. Returns (cleaned_text, match_count)."""
    total_matches = 0
    enabled_rules = sorted(
        [r for r in rules if r.get('enabled', True)],
        key=lambda r: r.get('priority', 999),
    )

    for rule in enabled_rules:
        pattern = rule.get('pattern', '')
        if not pattern:
            continue
        try:
            compiled = re.compile(pattern, re.MULTILINE)
        except re.error:
            continue

        action = rule.get('action', 'strip_line')

        if action == 'strip_line':
            lines = text.split('\n')
            new_lines = []
            for line in lines:
                if compiled.search(line):
                    total_matches += 1
                else:
                    new_lines.append(line)
            text = '\n'.join(new_lines)

        elif action == 'strip_match':
            matches = compiled.findall(text)
            total_matches += len(matches)
            text = compiled.sub('', text)

        elif action == 'replace':
            replacement = rule.get('replacement', '')
            matches = compiled.findall(text)
            total_matches += len(matches)
            text = compiled.sub(replacement, text)

        elif action == 'strip_paragraph':
            # Split into paragraphs (blocks separated by blank lines),
            # join each paragraph's lines into one string for matching,
            # remove the entire paragraph if it matches.
            paragraphs = re.split(r'\n\s*\n', text)
            kept = []
            for para in paragraphs:
                # Collapse internal line breaks + whitespace for matching
                joined = re.sub(r'\s+', ' ', para.strip())
                if compiled.search(joined):
                    total_matches += 1
                else:
                    kept.append(para)
            text = '\n\n'.join(kept)

    return text, total_matches


@NodeRegistry.register
class RegexRulesNode(BaseNode):
    node_type = 'regex_rules'
    label = 'Pattern Rules'
    category = 'text'
    description = 'Apply cleaning rules from a rule library — removes boilerplate, navigation, URLs, and other unwanted patterns'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text to clean')]
    outputs = [
        Port('cleaned', PortType.TEXT_BATCH, 'Cleaned text'),
        Port('metrics', PortType.METRICS, 'How many patterns matched', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'library': {
                'type': 'string',
                'title': 'Rule library',
                'description': 'Which set of cleaning rules to use',
                'default': 'nifty_archive',
            },
            'categories': {
                'type': 'string',
                'title': 'Rule categories',
                'description': 'Only apply rules from these categories (empty = all)',
                'default': '',
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        library_name = config.get('library', 'nifty_archive')
        category_filter = config.get('categories', '').strip()

        rules = load_rule_library(library_name)

        if category_filter:
            cats = {c.strip() for c in category_filter.split(',') if c.strip()}
            rules = [r for r in rules if r.get('category', '') in cats]

        out = []
        total_matches = 0

        for chunk in chunks:
            cleaned, matches = apply_rules(chunk.text, rules)
            total_matches += matches
            out.append(chunk.with_text(cleaned, 'regex_rules'))

        return {
            'cleaned': out,
            'out': out,  # backward compat
            'metrics': {
                'total': len(chunks),
                'total_rule_matches': total_matches,
                'rules_loaded': len(rules),
                'library': library_name,
            },
        }
