"""Marathon-length resumable trainer for the key-moment detector.

Design goals:
  * Runnable for 55+ hours
  * Stoppable at any time (SIGINT, kill, reboot, power loss) without data loss
  * Resumable from the latest checkpoint on relaunch with a single flag
  * Cross-platform: CUDA (orrion) and MPS (orrpheus) and CPU fallback

Checkpoint layout::

    <ckpt_dir>/
    ├── ckpt_000000500.pt   — step 500
    ├── ckpt_000001000.pt
    ├── ...
    ├── best.pt             — copy with lowest validation loss
    └── last.pt             — most recent (updated on SIGINT / exit)

Each .pt contains: {step, model, optimizer, scheduler, best_val_loss,
train_rng, torch_rng, cuda_rng, config}.

Relaunch with --resume picks up `last.pt` if present.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.video.dataset import (LABEL_TO_IDX, MultiModalDataset,
                                    collate_fn)
from training.video.model import KeyMomentDetector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    data_root: str = 'output/video_training'
    ckpt_dir: str = 'output/video_training/ckpt'

    # Schedule
    max_steps: int = 200_000      # ~55h at ~0.25s/step on 3070 / MPS
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Checkpoint / validation cadence
    save_every: int = 500          # steps between regular checkpoints
    val_every: int = 500           # steps between validation passes
    log_every: int = 20            # steps between stdout log lines
    keep_last_n: int = 5           # how many rolling ckpts to keep on disk

    # Misc
    seed: int = 42
    device: str = 'auto'           # 'auto' | 'cuda' | 'mps' | 'cpu'


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def pick_device(requested: str = 'auto') -> torch.device:
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None \
            and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_name(step: int) -> str:
    return f'ckpt_{step:010d}.pt'


def _find_latest(ckpt_dir: str) -> str | None:
    """Return path to the most recent checkpoint, preferring last.pt."""
    if not os.path.isdir(ckpt_dir):
        return None
    last = os.path.join(ckpt_dir, 'last.pt')
    if os.path.isfile(last):
        return last
    ckpts = [f for f in os.listdir(ckpt_dir)
             if f.startswith('ckpt_') and f.endswith('.pt')]
    if not ckpts:
        return None
    ckpts.sort()
    return os.path.join(ckpt_dir, ckpts[-1])


def _rotate_old(ckpt_dir: str, keep: int) -> None:
    """Delete old rolling ckpts, keeping only the most recent `keep`."""
    ckpts = sorted(f for f in os.listdir(ckpt_dir)
                   if f.startswith('ckpt_') and f.endswith('.pt'))
    for old in ckpts[:-keep]:
        try:
            os.remove(os.path.join(ckpt_dir, old))
        except OSError:
            pass


def save_checkpoint(path: str, *, step: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Any, best_val_loss: float,
                    config: TrainConfig) -> None:
    """Atomically write a checkpoint (write to .tmp, then rename)."""
    state = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_val_loss': best_val_loss,
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        'config': asdict(config),
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Any, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None and state.get('scheduler') is not None:
        scheduler.load_state_dict(state['scheduler'])

    try:
        random.setstate(state['python_rng'])
    except Exception:
        pass
    try:
        np.random.set_state(state['numpy_rng'])
    except Exception:
        pass
    try:
        torch.set_rng_state(state['torch_rng'])
    except Exception:
        pass
    if torch.cuda.is_available() and state.get('cuda_rng') is not None:
        try:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        except Exception:
            pass

    return {
        'step': int(state.get('step', 0)),
        'best_val_loss': float(state.get('best_val_loss', math.inf)),
    }


# ---------------------------------------------------------------------------
# Cosine LR schedule with warmup
# ---------------------------------------------------------------------------

class CosineWithWarmup:
    """Simple hand-rolled scheduler so its state survives checkpoints cleanly."""

    def __init__(self, optimizer: torch.optim.Optimizer, *,
                 warmup: int, total: int, base_lr: float,
                 min_lr_ratio: float = 0.01):
        self.optimizer = optimizer
        self.warmup = warmup
        self.total = total
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio
        self._step = 0

    def state_dict(self) -> dict[str, Any]:
        return {'step': self._step, 'warmup': self.warmup,
                'total': self.total, 'base_lr': self.base_lr,
                'min_lr': self.min_lr}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._step = int(state.get('step', 0))
        self.warmup = int(state.get('warmup', self.warmup))
        self.total = int(state.get('total', self.total))
        self.base_lr = float(state.get('base_lr', self.base_lr))
        self.min_lr = float(state.get('min_lr', self.min_lr))

    def get_lr(self) -> float:
        if self._step < self.warmup:
            return self.base_lr * (self._step + 1) / max(1, self.warmup)
        progress = (self._step - self.warmup) / max(1, self.total - self.warmup)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cos

    def step(self) -> float:
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self._step += 1
        return lr


# ---------------------------------------------------------------------------
# Train / validation loops
# ---------------------------------------------------------------------------

def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader,
                   loss_fn: nn.Module, device: torch.device,
                   max_batches: int | None = None) -> dict[str, float]:
    """Run a validation pass over the val loader and return metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    per_class_correct = [0] * len(LABEL_TO_IDX)
    per_class_seen = [0] * len(LABEL_TO_IDX)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device)
        logits = model(batch)
        loss = loss_fn(logits, batch['label'])
        preds = logits.argmax(dim=-1)

        total_loss += float(loss.item()) * batch['label'].size(0)
        total_correct += int((preds == batch['label']).sum().item())
        total_seen += batch['label'].size(0)
        for cls in range(len(LABEL_TO_IDX)):
            mask = batch['label'] == cls
            if mask.any():
                per_class_seen[cls] += int(mask.sum().item())
                per_class_correct[cls] += int((preds[mask] == cls).sum().item())

    model.train()
    if total_seen == 0:
        return {'loss': math.inf, 'acc': 0.0}

    out = {'loss': total_loss / total_seen,
           'acc': total_correct / total_seen}
    for cls, name in sorted(((v, k) for k, v in LABEL_TO_IDX.items())):
        seen = per_class_seen[cls]
        out[f'acc_{name}'] = per_class_correct[cls] / seen if seen else 0.0
    return out


def _infinite(loader: DataLoader):
    """Yield batches forever by cycling the DataLoader across epochs."""
    while True:
        for batch in loader:
            yield batch


def train(cfg: TrainConfig, resume: bool = False) -> None:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = pick_device(cfg.device)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    train_jsonl = os.path.join(cfg.data_root, 'train.jsonl')
    val_jsonl = os.path.join(cfg.data_root, 'val.jsonl')
    train_ds = MultiModalDataset(train_jsonl, root=cfg.data_root, train=True)
    val_ds = MultiModalDataset(val_jsonl, root=cfg.data_root, train=False)

    pin = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=pin, drop_last=True,
        persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=pin, persistent_workers=cfg.num_workers > 0)

    model = KeyMomentDetector().to(device)
    class_weights = train_ds.class_weights().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay)
    scheduler = CosineWithWarmup(
        optimizer, warmup=cfg.warmup_steps, total=cfg.max_steps,
        base_lr=cfg.learning_rate)

    start_step = 0
    best_val_loss = math.inf
    if resume:
        latest = _find_latest(cfg.ckpt_dir)
        if latest is not None:
            loaded = load_checkpoint(latest, model, optimizer, scheduler, device)
            start_step = loaded['step']
            best_val_loss = loaded['best_val_loss']
            print(f'[resume] loaded {latest} @ step {start_step}, '
                  f'best_val_loss={best_val_loss:.4f}', flush=True)
        else:
            print('[resume] no checkpoint found, starting from scratch',
                  flush=True)

    print(f'[setup] device={device} params={model.num_parameters():,} '
          f'train_n={len(train_ds)} val_n={len(val_ds)}', flush=True)
    print(f'[setup] class_weights={class_weights.tolist()}', flush=True)
    print(f'[setup] label_counts={train_ds.label_counts()}', flush=True)

    # ------------------------------------------------------------------
    # SIGINT handler — flush last.pt before exiting
    # ------------------------------------------------------------------
    stop_requested = {'flag': False}

    def _on_sigint(signum, frame):  # noqa: ARG001
        if stop_requested['flag']:
            print('\n[sigint] second interrupt, hard exit', flush=True)
            sys.exit(1)
        stop_requested['flag'] = True
        print('\n[sigint] stop requested, will flush after current step',
              flush=True)

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    model.train()
    step = start_step
    loss_sum = 0.0
    acc_sum = 0.0
    n_acc = 0
    step_start = time.time()

    try:
        for batch in _infinite(train_loader):
            if step >= cfg.max_steps or stop_requested['flag']:
                break

            batch = _to_device(batch, device)
            logits = model(batch)
            loss = loss_fn(logits, batch['label'])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            lr = scheduler.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc_sum += float((preds == batch['label']).float().mean().item())
                loss_sum += float(loss.item())
                n_acc += 1

            step += 1

            if step % cfg.log_every == 0:
                elapsed = time.time() - step_start
                its = cfg.log_every / max(elapsed, 1e-6)
                print(f'step {step}/{cfg.max_steps} | '
                      f'loss {loss_sum / n_acc:.4f} | '
                      f'acc {acc_sum / n_acc:.4f} | '
                      f'lr {lr:.2e} | '
                      f'{its:.2f} it/s', flush=True)
                loss_sum = 0.0
                acc_sum = 0.0
                n_acc = 0
                step_start = time.time()

            if step % cfg.val_every == 0:
                metrics = run_validation(model, val_loader, loss_fn, device)
                metrics_str = ' '.join(
                    f'{k}={v:.4f}' for k, v in metrics.items())
                print(f'[val step {step}] {metrics_str}', flush=True)
                if metrics['loss'] < best_val_loss:
                    best_val_loss = metrics['loss']
                    save_checkpoint(
                        os.path.join(cfg.ckpt_dir, 'best.pt'),
                        step=step, model=model, optimizer=optimizer,
                        scheduler=scheduler, best_val_loss=best_val_loss,
                        config=cfg)
                    print(f'[val step {step}] new best_val_loss={best_val_loss:.4f}',
                          flush=True)

            if step % cfg.save_every == 0:
                save_checkpoint(
                    os.path.join(cfg.ckpt_dir, _ckpt_name(step)),
                    step=step, model=model, optimizer=optimizer,
                    scheduler=scheduler, best_val_loss=best_val_loss,
                    config=cfg)
                save_checkpoint(
                    os.path.join(cfg.ckpt_dir, 'last.pt'),
                    step=step, model=model, optimizer=optimizer,
                    scheduler=scheduler, best_val_loss=best_val_loss,
                    config=cfg)
                _rotate_old(cfg.ckpt_dir, cfg.keep_last_n)
    finally:
        # Always flush last.pt on exit so we can resume cleanly next launch.
        save_checkpoint(
            os.path.join(cfg.ckpt_dir, 'last.pt'),
            step=step, model=model, optimizer=optimizer,
            scheduler=scheduler, best_val_loss=best_val_loss,
            config=cfg)
        print(f'[exit] step={step} best_val_loss={best_val_loss:.4f} '
              f'last.pt flushed', flush=True)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | None) -> TrainConfig:
    """Load a YAML or JSON config file, falling back to defaults."""
    if path is None:
        return TrainConfig()
    with open(path, encoding='utf-8') as f:
        raw = f.read()
    if path.endswith(('.yaml', '.yml')):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                'PyYAML is required to load YAML configs') from e
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw)
    # Keep only known fields so we tolerate extra keys for future expansion.
    known = {f for f in TrainConfig.__dataclass_fields__}
    return TrainConfig(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train the cbsr key-moment detector')
    parser.add_argument('--config', default=None,
                        help='Path to YAML/JSON config file')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in ckpt_dir')
    parser.add_argument('--data-root', default=None,
                        help='Override data_root from config')
    parser.add_argument('--ckpt-dir', default=None,
                        help='Override ckpt_dir from config')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Override max_steps from config')
    parser.add_argument('--device', default=None,
                        help='auto | cuda | mps | cpu')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.ckpt_dir is not None:
        cfg.ckpt_dir = args.ckpt_dir
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.device is not None:
        cfg.device = args.device
    train(cfg, resume=args.resume)


if __name__ == '__main__':
    main()
