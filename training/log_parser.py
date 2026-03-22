"""Parse training log output into structured metrics.

Supports:
- mlx-lm LoRA training output
- Unsloth/HuggingFace Trainer output
"""

import re
import json

# mlx-lm patterns
# "Iter 100: Train loss 2.345, Learning Rate 1.000e-04, It/sec 0.067, Tokens/sec 17.123, Trained Tokens 12345, Peak mem 12.345 GB"
_MLX_TRAIN = re.compile(
    r'Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+-]+),\s+'
    r'It/sec\s+([\d.]+),\s+Tokens/sec\s+([\d.]+),\s+Trained Tokens\s+(\d+),\s+'
    r'Peak mem\s+([\d.]+)\s+GB'
)
# "Iter 100: Val loss 2.123, Val took 45.678s"
_MLX_VAL = re.compile(
    r'Iter\s+(\d+):\s+Val loss\s+([\d.]+),\s+Val took\s+([\d.]+)s'
)
# "Iter 100: Saved adapter weights"
_MLX_SAVE = re.compile(r'Iter\s+(\d+):\s+Saved adapter')
# "Loading pretrained model" / "Fetching X pages"
_MLX_STATUS = re.compile(r'(Loading|Fetching|Starting|Generating)')

# Unsloth patterns
# {"loss": 2.345, "learning_rate": 1e-4, "epoch": 0.5}
_UNSLOTH_JSON = re.compile(r'^\s*\{.*"loss".*\}\s*$')
# tqdm: " 10%|██        | 100/1000 [01:23<12:34, 1.23it/s]"
_UNSLOTH_TQDM = re.compile(
    r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]'
)

# ANSI escape codes
_ANSI = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r')


def parse_line(line: str) -> dict | None:
    """Parse a single log line into a structured metric dict.

    Returns None if the line doesn't contain parseable metrics.
    Returns a dict with 'type' key indicating the metric type:
      - 'train': training loss + metrics
      - 'val': validation loss
      - 'save': checkpoint saved
      - 'status': informational message
      - 'progress': tqdm-style progress update
    """
    if not line or not line.strip():
        return None

    # Strip ANSI
    clean = _ANSI.sub('', line).strip() if '\x1b' in line or '\r' in line else line.strip()
    if not clean:
        return None

    # mlx-lm train line
    m = _MLX_TRAIN.match(clean)
    if m:
        return {
            'type': 'train',
            'iter': int(m.group(1)),
            'train_loss': float(m.group(2)),
            'lr': float(m.group(3)),
            'it_per_sec': float(m.group(4)),
            'tokens_per_sec': float(m.group(5)),
            'trained_tokens': int(m.group(6)),
            'peak_mem_gb': float(m.group(7)),
        }

    # mlx-lm val line
    m = _MLX_VAL.match(clean)
    if m:
        return {
            'type': 'val',
            'iter': int(m.group(1)),
            'val_loss': float(m.group(2)),
            'val_time_s': float(m.group(3)),
        }

    # mlx-lm checkpoint save
    m = _MLX_SAVE.match(clean)
    if m:
        return {
            'type': 'save',
            'iter': int(m.group(1)),
        }

    # Unsloth JSON log
    if _UNSLOTH_JSON.match(clean):
        try:
            data = json.loads(clean)
            return {
                'type': 'train',
                'train_loss': data.get('loss'),
                'lr': data.get('learning_rate'),
                'epoch': data.get('epoch'),
                'iter': data.get('step'),
            }
        except json.JSONDecodeError:
            pass

    # Unsloth tqdm progress
    m = _UNSLOTH_TQDM.search(clean)
    if m:
        return {
            'type': 'progress',
            'pct': int(m.group(1)),
            'current': int(m.group(2)),
            'total': int(m.group(3)),
            'timing': m.group(4),
        }

    # Status messages
    if _MLX_STATUS.match(clean):
        return {
            'type': 'status',
            'message': clean,
        }

    return None


def estimate_eta(metrics: list, total_iters: int) -> float | None:
    """Estimate remaining time in seconds based on recent iteration speed.

    Uses the last 20 train metrics to calculate average it/sec.
    """
    train_metrics = [m for m in metrics if m.get('type') == 'train' and m.get('it_per_sec')]
    if not train_metrics:
        return None

    recent = train_metrics[-20:]
    avg_speed = sum(m['it_per_sec'] for m in recent) / len(recent)
    if avg_speed <= 0:
        return None

    current_iter = recent[-1].get('iter', 0)
    remaining = total_iters - current_iter
    if remaining <= 0:
        return 0.0

    return remaining / avg_speed


def downsample_metrics(metrics: list, max_points: int = 500) -> list:
    """Downsample metrics for chart rendering.

    Keeps first, last, and evenly spaced points.
    Always preserves val loss points.
    """
    if len(metrics) <= max_points:
        return metrics

    # Separate val metrics (always keep) and train metrics (downsample)
    val_metrics = [m for m in metrics if m.get('type') == 'val']
    train_metrics = [m for m in metrics if m.get('type') == 'train']

    remaining_budget = max_points - len(val_metrics)
    if remaining_budget <= 2:
        return val_metrics[:max_points]

    # Evenly sample train metrics
    step = len(train_metrics) / remaining_budget
    sampled = []
    for i in range(remaining_budget):
        idx = int(i * step)
        if idx < len(train_metrics):
            sampled.append(train_metrics[idx])

    # Merge and sort by iteration
    combined = sampled + val_metrics
    combined.sort(key=lambda m: m.get('iter', 0))
    return combined
