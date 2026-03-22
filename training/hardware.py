"""Cross-platform hardware detection for Orracle Trainer.

Detects CPU, RAM, GPU (NVIDIA + Apple Silicon), ML frameworks,
disk space, and available tools (ollama, ffmpeg, etc.).
"""

import os
import json
import platform
import shutil
import subprocess
import importlib


def detect_hardware() -> dict:
    """Detect all hardware capabilities of the current machine."""
    hw = {
        'hostname': platform.node(),
        'platform': platform.system(),
        'arch': platform.machine(),
        'cpu': _detect_cpu(),
        'ram': _detect_ram(),
        'gpu': _detect_gpu(),
        'frameworks': _detect_frameworks(),
        'disk': _detect_disk(),
        'tools': _detect_tools(),
    }
    hw['capabilities'] = _derive_capabilities(hw)
    return hw


def _detect_cpu() -> dict:
    info = {
        'cores': os.cpu_count(),
        'arch': platform.machine(),
    }
    # Try to get model name
    if platform.system() == 'Darwin':
        try:
            r = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                info['model'] = r.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Apple Silicon chip name
        try:
            r = subprocess.run(['sysctl', '-n', 'hw.chip'],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                info['chip'] = r.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    elif platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        info['model'] = line.split(':', 1)[1].strip()
                        break
        except FileNotFoundError:
            pass
    return info


def _detect_ram() -> dict:
    """Detect total and available RAM in GB."""
    info = {}
    if platform.system() == 'Darwin':
        try:
            r = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                total_bytes = int(r.stdout.strip())
                info['total_gb'] = round(total_bytes / (1024**3), 1)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
    elif platform.system() == 'Linux':
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        info['total_gb'] = round(kb / (1024**2), 1)
                    elif line.startswith('MemAvailable:'):
                        kb = int(line.split()[1])
                        info['available_gb'] = round(kb / (1024**2), 1)
        except FileNotFoundError:
            pass

    # psutil fallback for available RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        if 'total_gb' not in info:
            info['total_gb'] = round(mem.total / (1024**3), 1)
        info['available_gb'] = round(mem.available / (1024**3), 1)
        info['used_pct'] = mem.percent
    except ImportError:
        pass

    return info


def _detect_gpu() -> dict:
    """Detect NVIDIA GPU via nvidia-smi or Apple Silicon GPU."""
    info = {}

    # NVIDIA
    try:
        r = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total,memory.free,driver_version,compute_cap',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            for i, line in enumerate(r.stdout.strip().split('\n')):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu = {
                        'name': parts[0],
                        'vram_total_mb': int(float(parts[1])),
                        'vram_free_mb': int(float(parts[2])),
                        'driver': parts[3],
                        'type': 'nvidia',
                    }
                    if len(parts) >= 5:
                        gpu['compute_cap'] = parts[4]
                    info[f'gpu_{i}'] = gpu
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Apple Silicon — unified memory is shared GPU/CPU
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            r = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType', '-json'],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                data = json.loads(r.stdout)
                displays = data.get('SPDisplaysDataType', [])
                for d in displays:
                    name = d.get('sppci_model', 'Apple GPU')
                    cores = d.get('sppci_cores', '')
                    info['gpu_0'] = {
                        'name': name,
                        'cores': cores,
                        'type': 'apple_silicon',
                        'unified_memory': True,
                    }
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

    return info


def _detect_frameworks() -> dict:
    """Check which ML frameworks are installed."""
    frameworks = {}

    # PyTorch
    try:
        import torch
        frameworks['torch'] = {
            'version': torch.__version__,
            'cuda': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
    except ImportError:
        frameworks['torch'] = None

    # MLX
    try:
        import mlx.core as mx
        frameworks['mlx'] = {
            'version': getattr(mx, '__version__', 'unknown'),
        }
    except ImportError:
        frameworks['mlx'] = None

    # mlx-lm
    try:
        mlx_lm = importlib.import_module('mlx_lm')
        frameworks['mlx_lm'] = {
            'version': getattr(mlx_lm, '__version__', 'installed'),
        }
    except ImportError:
        frameworks['mlx_lm'] = None

    # Unsloth
    try:
        import unsloth
        frameworks['unsloth'] = {
            'version': getattr(unsloth, '__version__', 'installed'),
        }
    except ImportError:
        frameworks['unsloth'] = None

    # Transformers
    try:
        import transformers
        frameworks['transformers'] = {
            'version': transformers.__version__,
        }
    except ImportError:
        frameworks['transformers'] = None

    return frameworks


def _detect_disk() -> dict:
    """Get disk usage for common paths."""
    info = {}
    for name, path in [('home', os.path.expanduser('~')), ('root', '/')]:
        try:
            usage = shutil.disk_usage(path)
            info[name] = {
                'total_gb': round(usage.total / (1024**3), 1),
                'free_gb': round(usage.free / (1024**3), 1),
                'used_pct': round(usage.used / usage.total * 100, 1),
            }
        except OSError:
            pass
    return info


def _detect_tools() -> dict:
    """Check for commonly needed CLI tools."""
    tools = {}
    for tool in ['ollama', 'ffmpeg', 'python3', 'ssh', 'git']:
        path = shutil.which(tool)
        if path:
            tools[tool] = path
    return tools


def _derive_capabilities(hw: dict) -> list:
    """Derive training capabilities from detected hardware."""
    caps = []
    fw = hw.get('frameworks', {})

    # MLX LoRA — requires Apple Silicon + mlx-lm
    if fw.get('mlx_lm') and hw.get('gpu', {}).get('gpu_0', {}).get('type') == 'apple_silicon':
        caps.append('mlx_lora')

    # Unsloth QLoRA — requires NVIDIA CUDA + unsloth
    if fw.get('unsloth') and fw.get('torch', {}) and fw['torch'].get('cuda'):
        caps.append('unsloth_qlora')

    # Transformers SFTTrainer — just needs transformers + torch
    if fw.get('transformers') and fw.get('torch'):
        caps.append('transformers_sft')

    # Ollama inference
    if hw.get('tools', {}).get('ollama'):
        caps.append('ollama_serve')

    return caps


def format_gpu_summary(hw: dict) -> str:
    """Human-readable GPU summary."""
    gpu = hw.get('gpu', {})
    if not gpu:
        return 'No GPU detected'
    g = gpu.get('gpu_0', {})
    name = g.get('name', 'Unknown')
    if g.get('type') == 'nvidia':
        vram = g.get('vram_total_mb', 0)
        return f'{name} ({vram} MB VRAM)'
    elif g.get('type') == 'apple_silicon':
        cores = g.get('cores', '?')
        return f'{name} ({cores} cores, unified memory)'
    return name


def format_ram_summary(hw: dict) -> str:
    """Human-readable RAM summary."""
    ram = hw.get('ram', {})
    total = ram.get('total_gb', '?')
    avail = ram.get('available_gb')
    if avail is not None:
        return f'{total} GB total, {avail} GB available'
    return f'{total} GB total'
