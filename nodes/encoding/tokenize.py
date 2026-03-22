"""Tokenization node — tiktoken/sentencepiece wrapper."""

from __future__ import annotations

from nodes.base import BaseNode, DataChunk, NodeRegistry, Port, PortType


@NodeRegistry.register
class TokenizeNode(BaseNode):
    node_type = 'tokenize'
    label = 'Tokenize'
    category = 'encoding'
    description = 'Count tokens per chunk — useful for estimating training cost and verifying chunk sizes'
    inputs = [Port('text', PortType.TEXT_BATCH, 'Text to count tokens for')]
    outputs = [
        Port('counted', PortType.TEXT_BATCH, 'Same text, with token counts added to metadata'),
        Port('metrics', PortType.METRICS, 'Token stats', required=False),
    ]
    params_schema = {
        'type': 'object',
        'properties': {
            'encoding': {
                'type': 'string',
                'title': 'Tokenizer',
                'enum': ['cl100k_base', 'o200k_base', 'p50k_base', 'gpt2'],
                'default': 'cl100k_base',
            },
        },
    }

    def process(self, inputs, config):
        chunks = inputs.get('text', inputs.get('in', []))
        enc_name = config.get('encoding', 'cl100k_base')

        try:
            import tiktoken
            enc = tiktoken.get_encoding(enc_name)
        except ImportError:
            # Fallback: estimate ~4 chars per token
            enc = None

        out = []
        total_tokens = 0
        total_chars = 0

        for chunk in chunks:
            if enc:
                tokens = enc.encode(chunk.text)
                token_count = len(tokens)
            else:
                token_count = len(chunk.text) // 4

            total_tokens += token_count
            total_chars += len(chunk.text)

            new_chunk = DataChunk(
                text=chunk.text,
                metadata={**chunk.metadata, 'token_count': token_count},
                history=[*chunk.history, 'tokenize'],
            )
            out.append(new_chunk)

        return {
            'counted': out,
            'out': out,
            'metrics': {
                'total_chunks': len(chunks),
                'total_tokens': total_tokens,
                'total_chars': total_chars,
                'avg_tokens': round(total_tokens / max(1, len(chunks)), 1),
                'chars_per_token': round(total_chars / max(1, total_tokens), 2),
                'tokenizer': enc_name,
                'tiktoken_available': enc is not None,
            },
        }
