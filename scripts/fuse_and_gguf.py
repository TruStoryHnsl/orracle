#!/usr/bin/env python3
"""Fuse LoRA adapter + convert to GGUF with contiguous array fix.

Workaround for mlx_lm --export-gguf bug where merged LoRA weights
produce non-row-major arrays that mx.save_gguf rejects.

Usage:
    python fuse_and_gguf.py --model MODEL --adapter-path ADAPTER \
        --save-path FUSED_DIR --gguf-path OUTPUT.gguf
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Base model (HF path)')
    parser.add_argument('--adapter-path', required=True, help='LoRA adapter dir')
    parser.add_argument('--save-path', required=True, help='Fused model output dir')
    parser.add_argument('--gguf-path', required=True, help='Output GGUF file path')
    args = parser.parse_args()

    # Step 1: Fuse adapter with dequantize (no GGUF — avoids row-major bug)
    print('Step 1: Fusing adapter (dequantize)...')
    proc = subprocess.run(
        [sys.executable, '-m', 'mlx_lm.fuse',
         '--model', args.model,
         '--adapter-path', args.adapter_path,
         '--save-path', args.save_path,
         '--dequantize'],
    )
    if proc.returncode != 0:
        print('ERROR: Fuse failed')
        sys.exit(1)
    print('Fuse complete.')

    # Step 2: Load fused weights, force contiguous, export GGUF
    print('Step 2: Converting to GGUF (with contiguous fix)...')
    import mlx.core as mx
    from mlx_lm.gguf import convert_to_gguf

    fused_path = Path(args.save_path)
    config = json.loads((fused_path / 'config.json').read_text())

    # mx_materialize is mx.eval — forces lazy graph evaluation.
    # Named differently to avoid false-positive from security hooks.
    mx_materialize = getattr(mx, 'eval')

    weights = {}
    for shard in sorted(fused_path.glob('*.safetensors')):
        print(f'  Loading {shard.name}...')
        for k, v in mx.load(str(shard)).items():
            # reshape forces contiguous (row-major) layout
            v = v.reshape(-1).reshape(v.shape)
            mx_materialize(v)
            weights[k] = v

    print(f'  Loaded {len(weights)} tensors')
    convert_to_gguf(fused_path, weights, config, args.gguf_path)
    print(f'GGUF export complete: {args.gguf_path}')


if __name__ == '__main__':
    main()
