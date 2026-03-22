"""Training job lifecycle management.

Handles: config generation, subprocess start/monitor/stop,
metric collection, job persistence.
"""

import os
import re
import signal
import subprocess
import threading
import time
import yaml

from . import log_parser

# ---------------------------------------------------------------------------
# Job storage
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
JOBS_FILE = os.path.join(CONFIG_DIR, 'jobs.yaml')

_jobs_lock = threading.Lock()


def _load_jobs() -> dict:
    try:
        with open(JOBS_FILE) as f:
            data = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        data = {}
    return data.get('jobs', {}) if isinstance(data.get('jobs'), dict) else {}


def _save_jobs(jobs: dict):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(JOBS_FILE, 'w') as f:
        yaml.dump({'jobs': jobs}, f, default_flow_style=False, sort_keys=False)


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _load_jobs().get(job_id)


def list_jobs() -> list:
    with _jobs_lock:
        jobs = _load_jobs()
    return sorted(jobs.values(), key=lambda j: j.get('started', ''), reverse=True)


def update_job(job_id: str, updates: dict):
    with _jobs_lock:
        jobs = _load_jobs()
        if job_id in jobs:
            jobs[job_id].update(updates)
            _save_jobs(jobs)


# ---------------------------------------------------------------------------
# Active job tracking (in-memory, per-process)
# ---------------------------------------------------------------------------

_active_jobs = {}  # job_id -> {process, output_lines, metrics, done, exit_code}
_active_lock = threading.Lock()


def get_active_job(job_id: str) -> dict | None:
    with _active_lock:
        return _active_jobs.get(job_id)


def get_all_active() -> dict:
    with _active_lock:
        return {jid: {'done': j['done'], 'exit_code': j['exit_code'],
                       'lines': len(j['output_lines']),
                       'metrics': len(j['metrics'])}
                for jid, j in _active_jobs.items()}


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

# Default LoRA targets for Mistral/Llama architectures
DEFAULT_LORA_KEYS = [
    'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj',
    'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj',
]

# Base model presets
MODEL_PRESETS = {
    'mistral-7b-4bit': {
        'model': 'mlx-community/Mistral-7B-Instruct-v0.3-4bit',
        'label': 'Mistral 7B v0.3 (4-bit)',
    },
    'llama-3.1-8b': {
        'model': 'mlx-community/Meta-Llama-3.1-8B-Instruct-4bit',
        'label': 'Llama 3.1 8B (4-bit)',
    },
    'qwen-2.5-7b': {
        'model': 'mlx-community/Qwen2.5-7B-Instruct-4bit',
        'label': 'Qwen 2.5 7B (4-bit)',
    },
}


def generate_mlx_config(params: dict) -> str:
    """Generate an mlx-lm LoRA YAML config from form parameters.

    Returns the path to the written config file.
    """
    job_id = params['job_id']

    config = {
        'model': params.get('model', MODEL_PRESETS['mistral-7b-4bit']['model']),
        'train': True,
        'data': params.get('data_path', './data/mlx'),
        'adapter_path': params.get('adapter_path', f'adapters_{job_id}'),
        'iters': int(params.get('iters', 25000)),
        'batch_size': int(params.get('batch_size', 2)),
        'grad_checkpoint': True,
        'num_layers': int(params.get('num_layers', 32)),
        'lora_parameters': {
            'keys': params.get('lora_keys', DEFAULT_LORA_KEYS),
            'rank': int(params.get('lora_rank', 16)),
            'scale': float(params.get('lora_scale', 2.0)),
            'dropout': float(params.get('lora_dropout', 0.0)),
        },
        'learning_rate': float(params.get('lr', 2e-4)),
        'lr_schedule': {
            'name': 'cosine_decay',
            'warmup': int(params.get('warmup', 500)),
            'warmup_init': 1e-6,
            'arguments': [
                float(params.get('lr', 2e-4)),         # init (start LR after warmup)
                int(params.get('iters', 25000)),        # decay_steps
                float(params.get('lr_end', 1e-5)),      # end (absolute final LR)
            ],
        },
        'steps_per_eval': int(params.get('eval_every', 500)),
        'val_batches': int(params.get('val_batches', 50)),
        'save_every': int(params.get('save_every', 200)),
        'max_seq_length': int(params.get('max_seq_len', 2048)),
        'seed': 42,
    }

    # Resume from checkpoint
    if params.get('resume_adapter'):
        config['resume_adapter_file'] = params['resume_adapter']

    config_path = os.path.join(CONFIG_DIR, f'job_{job_id}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config_path


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

_ansi_re = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r')


def start_job(params: dict) -> str:
    """Start a training job. Returns job_id."""
    job_id = params.get('job_id') or str(int(time.time()))
    params['job_id'] = job_id

    framework = params.get('framework', 'mlx_lora')
    machine = params.get('machine', 'local')
    work_dir = params.get('work_dir', os.path.expanduser('~/projects/orrapus/niftytune'))

    if framework == 'mlx_lora':
        config_path = generate_mlx_config(params)
        cmd = ['python', '-u', '-m', 'mlx_lm', 'lora', '--config', config_path]
    elif framework == 'unsloth_qlora':
        cmd = _build_unsloth_cmd(params)
    else:
        raise ValueError(f'Unknown framework: {framework}')

    # If remote, wrap in SSH
    if machine != 'local':
        hostname = params.get('hostname', machine)
        cmd = ['ssh', '-tt', hostname, f'cd {work_dir} && ' + ' '.join(cmd)]

    # Persist job metadata
    job_meta = {
        'id': job_id,
        'status': 'starting',
        'framework': framework,
        'machine': machine,
        'model': params.get('model', ''),
        'output_name': params.get('output_name', f'model_{job_id}'),
        'total_iters': int(params.get('iters', 25000)),
        'config': params.get('_config_snapshot', {}),
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': None,
        'final_loss': None,
        'best_val_loss': None,
    }
    with _jobs_lock:
        jobs = _load_jobs()
        jobs[job_id] = job_meta
        _save_jobs(jobs)

    # Start subprocess
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=work_dir,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        update_job(job_id, {'status': 'failed', 'error': str(e),
                            'finished': time.strftime('%Y-%m-%d %H:%M:%S')})
        raise

    # Track in memory
    active = {
        'process': proc,
        'output_lines': [],
        'metrics': [],
        'done': False,
        'exit_code': 0,
        'total_iters': int(params.get('iters', 25000)),
        'best_val_loss': None,
    }
    with _active_lock:
        _active_jobs[job_id] = active

    update_job(job_id, {'status': 'running', 'pid': proc.pid})

    # Background reader thread
    t = threading.Thread(target=_read_output, args=(job_id,), daemon=True)
    t.start()

    return job_id


def _build_unsloth_cmd(params: dict) -> list:
    """Build command line for Unsloth training."""
    cmd = ['python', '-u', 'train_cpt.py']
    if params.get('model'):
        cmd.extend(['--model', params['model']])
    if params.get('iters'):
        cmd.extend(['--epochs', str(params.get('epochs', 3))])
    if params.get('batch_size'):
        cmd.extend(['--batch-size', str(params['batch_size'])])
    if params.get('lora_rank'):
        cmd.extend(['--lora-r', str(params['lora_rank'])])
    if params.get('max_seq_len'):
        cmd.extend(['--max-seq-len', str(params['max_seq_len'])])
    if params.get('lr'):
        cmd.extend(['--lr', str(params['lr'])])
    return cmd


def _read_output(job_id: str):
    """Background thread: read subprocess stdout, parse metrics."""
    active = _active_jobs.get(job_id)
    if not active:
        return

    proc = active['process']
    best_val = None

    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            clean = _ansi_re.sub('', line) if '\x1b' in line or '\r' in line else line
            if not clean.strip():
                continue

            with _active_lock:
                active['output_lines'].append(clean)
                if len(active['output_lines']) > 10_000:
                    del active['output_lines'][:5_000]

            # Parse metrics
            metric = log_parser.parse_line(clean)
            if metric:
                with _active_lock:
                    active['metrics'].append(metric)

                # Track best val loss
                if metric.get('type') == 'val' and metric.get('val_loss') is not None:
                    if best_val is None or metric['val_loss'] < best_val:
                        best_val = metric['val_loss']
                        active['best_val_loss'] = best_val

        proc.wait()
    except Exception:
        pass

    exit_code = proc.returncode or 0
    with _active_lock:
        active['done'] = True
        active['exit_code'] = exit_code

    # Determine final status
    status = 'completed' if exit_code == 0 else 'failed'

    # Get final train loss
    final_loss = None
    with _active_lock:
        train_metrics = [m for m in active['metrics'] if m.get('type') == 'train']
        if train_metrics:
            final_loss = train_metrics[-1].get('train_loss')

    update_job(job_id, {
        'status': status,
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        'exit_code': exit_code,
        'final_loss': final_loss,
        'best_val_loss': best_val,
    })

    # Evict from memory after grace period (let SSE clients drain)
    def _evict():
        time.sleep(300)
        with _active_lock:
            _active_jobs.pop(job_id, None)
    threading.Thread(target=_evict, daemon=True).start()


def stop_job(job_id: str, force: bool = False) -> bool:
    """Stop a running job. Returns True if signal was sent."""
    with _active_lock:
        active = _active_jobs.get(job_id)
        if not active or active['done']:
            return False
        proc = active['process']

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(proc.pid, sig)
    except ProcessLookupError:
        pass

    if not force:
        # Wait 10s then force kill
        def _force_kill():
            time.sleep(10)
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        threading.Thread(target=_force_kill, daemon=True).start()

    update_job(job_id, {'status': 'cancelled',
                        'finished': time.strftime('%Y-%m-%d %H:%M:%S')})
    return True
