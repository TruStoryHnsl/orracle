"""Remote machine management via SSH.

Handles: SSH connectivity, remote hardware detection, remote training
start/stop/monitor, log streaming over SSH, Wake-on-LAN, and job queuing
for offline machines.
"""

import os
import re
import json
import signal
import socket
import struct
import subprocess
import threading
import time

import yaml

from . import log_parser

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')

# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh_cmd(hostname: str, command: str, timeout: int = 10) -> dict:
    """Run a command on a remote machine via SSH.

    Returns {'ok': bool, 'stdout': str, 'stderr': str, 'returncode': int}.
    """
    try:
        r = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
             hostname, command],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            'ok': r.returncode == 0,
            'stdout': r.stdout,
            'stderr': r.stderr,
            'returncode': r.returncode,
        }
    except subprocess.TimeoutExpired:
        return {'ok': False, 'stdout': '', 'stderr': 'SSH timeout', 'returncode': -1}
    except FileNotFoundError:
        return {'ok': False, 'stdout': '', 'stderr': 'ssh not found', 'returncode': -1}


def _detect_remote_frameworks(hostname: str) -> dict:
    """Detect ML frameworks on a remote machine by probing imports."""
    # Check each framework individually for robustness
    frameworks = {}
    checks = [
        ('mlx', 'import mlx.core as mx; print(getattr(mx,"__version__","yes"))'),
        ('mlx_lm', 'import mlx_lm; print(getattr(mlx_lm,"__version__","yes"))'),
        ('torch', 'import torch; print(torch.__version__)'),
        ('unsloth', 'import unsloth; print("yes")'),
        ('transformers', 'import transformers; print(transformers.__version__)'),
    ]
    # Find the right python — try venv first (use ls to handle ~ expansion)
    python_cmd = None
    for candidate in ['~/niftytune/venv_mlx/bin/python3',
                      '~/niftytune/venv/bin/python3', 'python3']:
        r = _ssh_cmd(hostname, f'ls {candidate} 2>/dev/null && echo found')
        if r['ok'] and 'found' in r['stdout']:
            python_cmd = candidate
            break
    if not python_cmd:
        return {'ok': False}

    for name, check in checks:
        r = _ssh_cmd(hostname, f"{python_cmd} -c '{check}' 2>/dev/null", timeout=15)
        if r['ok'] and r['stdout'].strip():
            frameworks[name] = r['stdout'].strip()

    return {'ok': True, 'frameworks': frameworks, 'python': python_cmd}


def test_connection(hostname: str) -> dict:
    """Test SSH connectivity to a remote machine."""
    result = _ssh_cmd(hostname, 'echo ok')
    return {
        'reachable': result['ok'] and 'ok' in result['stdout'],
        'error': result['stderr'].strip() if not result['ok'] else None,
    }


def upload_file(hostname: str, local_path: str, remote_path: str) -> dict:
    """Upload a file to a remote machine via SCP."""
    try:
        r = subprocess.run(
            ['scp', '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
             local_path, f'{hostname}:{remote_path}'],
            capture_output=True, text=True, timeout=30,
        )
        return {'ok': r.returncode == 0, 'stderr': r.stderr}
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return {'ok': False, 'stderr': str(e)}


# ---------------------------------------------------------------------------
# Remote hardware detection
# ---------------------------------------------------------------------------

def detect_remote_hardware(hostname: str) -> dict:
    """Detect hardware on a remote machine via SSH."""
    hw = {'hostname': '', 'platform': '', 'arch': '', 'cpu': {}, 'ram': {},
           'gpu': {}, 'frameworks': {}, 'disk': {}, 'tools': {},
           'capabilities': [], '_remote': True}

    # Basic info
    r = _ssh_cmd(hostname, 'hostname && uname -s && uname -m', timeout=10)
    if not r['ok']:
        return {**hw, '_error': r['stderr']}
    lines = r['stdout'].strip().split('\n')
    if len(lines) >= 3:
        hw['hostname'] = lines[0]
        hw['platform'] = lines[1]
        hw['arch'] = lines[2]

    # CPU
    if hw['platform'] == 'Darwin':
        # Apple Silicon doesn't have machdep.cpu.brand_string
        r = _ssh_cmd(hostname,
                     'echo $(sysctl -n hw.chip 2>/dev/null || '
                     'sysctl -n machdep.cpu.brand_string 2>/dev/null || echo Unknown); '
                     'sysctl -n hw.ncpu 2>/dev/null')
        if r['ok']:
            parts = r['stdout'].strip().split('\n')
            hw['cpu'] = {
                'model': parts[0].strip() if parts else '',
                'cores': int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 0,
            }
    else:
        r = _ssh_cmd(hostname,
                     'grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 && '
                     'nproc 2>/dev/null')
        if r['ok']:
            parts = r['stdout'].strip().split('\n')
            hw['cpu'] = {
                'model': parts[0].strip() if parts else '',
                'cores': int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
            }

    # RAM
    if hw['platform'] == 'Darwin':
        r = _ssh_cmd(hostname, 'sysctl -n hw.memsize')
        if r['ok']:
            try:
                total = int(r['stdout'].strip())
                hw['ram'] = {'total_gb': round(total / (1024**3), 1)}
            except ValueError:
                pass
    else:
        r = _ssh_cmd(hostname, 'grep MemTotal /proc/meminfo | awk \'{print $2}\'')
        if r['ok']:
            try:
                kb = int(r['stdout'].strip())
                hw['ram'] = {'total_gb': round(kb / (1024**2), 1)}
            except ValueError:
                pass

    # GPU — Apple Silicon
    if hw['platform'] == 'Darwin' and hw['arch'] == 'arm64':
        r = _ssh_cmd(hostname,
                     'system_profiler SPDisplaysDataType -json 2>/dev/null',
                     timeout=15)
        if r['ok']:
            try:
                data = json.loads(r['stdout'])
                displays = data.get('SPDisplaysDataType', [])
                for d in displays:
                    hw['gpu']['gpu_0'] = {
                        'name': d.get('sppci_model', 'Apple GPU'),
                        'cores': d.get('sppci_cores', ''),
                        'type': 'apple_silicon',
                        'unified_memory': True,
                    }
            except json.JSONDecodeError:
                pass

    # GPU — NVIDIA
    r = _ssh_cmd(hostname,
                 'nvidia-smi --query-gpu=name,memory.total,memory.free '
                 '--format=csv,noheader,nounits 2>/dev/null')
    if r['ok'] and r['stdout'].strip():
        for i, line in enumerate(r['stdout'].strip().split('\n')):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                hw['gpu'][f'gpu_{i}'] = {
                    'name': parts[0],
                    'vram_total_mb': int(float(parts[1])),
                    'vram_free_mb': int(float(parts[2])),
                    'type': 'nvidia',
                }

    # Frameworks — detect via venv python, then system python
    fw_result = _detect_remote_frameworks(hostname)
    if fw_result.get('ok'):
        for k, v in fw_result.get('frameworks', {}).items():
            hw['frameworks'][k] = {'version': v}

    # Disk
    r = _ssh_cmd(hostname, 'df -k ~ | tail -1')
    if r['ok']:
        parts = r['stdout'].split()
        if len(parts) >= 4:
            try:
                total_kb = int(parts[1])
                used_kb = int(parts[2])
                free_kb = int(parts[3])
                hw['disk']['home'] = {
                    'total_gb': round(total_kb / (1024**2), 1),
                    'free_gb': round(free_kb / (1024**2), 1),
                    'used_pct': round(used_kb / total_kb * 100, 1) if total_kb else 0,
                }
            except ValueError:
                pass

    # Tools
    for tool in ['ollama', 'python3', 'git', 'screen']:
        r = _ssh_cmd(hostname, f'which {tool} 2>/dev/null')
        if r['ok'] and r['stdout'].strip():
            hw['tools'][tool] = r['stdout'].strip()

    # Capabilities
    caps = []
    if hw['frameworks'].get('mlx_lm') and hw['gpu'].get('gpu_0', {}).get('type') == 'apple_silicon':
        caps.append('mlx_lora')
    if hw['frameworks'].get('unsloth') and hw['frameworks'].get('torch'):
        caps.append('unsloth_qlora')
    if hw['frameworks'].get('transformers') and hw['frameworks'].get('torch'):
        caps.append('transformers_sft')
    if hw['tools'].get('ollama'):
        caps.append('ollama_serve')
    hw['capabilities'] = caps

    return hw


# ---------------------------------------------------------------------------
# Service discovery
# ---------------------------------------------------------------------------

# Known services with their default ports and health check paths
KNOWN_SERVICES = {
    'comfyui': {'port': 8188, 'path': '/system_stats'},
    'ollama': {'port': 11434, 'path': '/api/version'},
    'orrapus': {'port': 5000, 'path': '/'},
    'orrbit': {'port': 5001, 'path': '/'},
}


def probe_services(hostname: str, services: dict = None) -> dict:
    """Probe a remote machine for running services.

    Args:
        hostname: SSH hostname (or 'localhost' for local)
        services: Optional dict of {name: {url, port}} from machines.yaml.
                  If not provided, probes KNOWN_SERVICES defaults.

    Returns:
        {service_name: {'available': bool, 'url': str, 'port': int}}
    """
    targets = services or {k: {'port': v['port']} for k, v in KNOWN_SERVICES.items()}
    results = {}

    for name, svc in targets.items():
        port = svc.get('port', KNOWN_SERVICES.get(name, {}).get('port'))
        if not port:
            continue
        url = svc.get('url', f'http://localhost:{port}')
        health_path = KNOWN_SERVICES.get(name, {}).get('path', '/')

        if hostname in ('localhost', '127.0.0.1') or hostname == '':
            # Local check — direct HTTP
            import urllib.request
            import urllib.error
            try:
                req = urllib.request.Request(f'{url}{health_path}')
                with urllib.request.urlopen(req, timeout=3) as resp:
                    results[name] = {'available': True, 'url': url, 'port': port}
            except (urllib.error.URLError, OSError):
                results[name] = {'available': False, 'url': url, 'port': port}
        else:
            # Remote check — curl via SSH
            r = _ssh_cmd(hostname,
                         f'curl -s -o /dev/null -w "%{{http_code}}" '
                         f'http://localhost:{port}{health_path} 2>/dev/null',
                         timeout=8)
            available = r['ok'] and r['stdout'].strip() not in ('', '000')
            results[name] = {'available': available, 'url': url, 'port': port}

    return results


# ---------------------------------------------------------------------------
# Remote training detection (find running jobs)
# ---------------------------------------------------------------------------

def detect_remote_training(hostname: str, niftytune_dir: str = '~/niftytune') -> list:
    """Detect running training processes on a remote machine."""
    jobs = []

    # Check for mlx_lm processes
    r = _ssh_cmd(hostname, 'ps aux | grep "[m]lx_lm.*lora" 2>/dev/null')
    if r['ok'] and r['stdout'].strip():
        for line in r['stdout'].strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                pid = parts[1]
                cmd = ' '.join(parts[10:])
                # Extract config file
                config = ''
                if '--config' in cmd:
                    idx = cmd.index('--config') + len('--config')
                    config = cmd[idx:].strip().split()[0] if idx < len(cmd) else ''
                jobs.append({
                    'pid': pid,
                    'command': cmd,
                    'config': config,
                    'framework': 'mlx_lora',
                })

    # Check for Unsloth/transformers processes
    r = _ssh_cmd(hostname, 'ps aux | grep "[t]rain_cpt\\|[t]rain_sft" 2>/dev/null')
    if r['ok'] and r['stdout'].strip():
        for line in r['stdout'].strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                jobs.append({
                    'pid': parts[1],
                    'command': ' '.join(parts[10:]),
                    'framework': 'unsloth',
                })

    # Check screen sessions
    r = _ssh_cmd(hostname, 'screen -ls 2>/dev/null')
    if r['ok']:
        for line in r['stdout'].split('\n'):
            m = re.search(r'(\d+)\.(\S+)\s+\((\w+)\)', line)
            if m:
                for job in jobs:
                    job['screen_session'] = m.group(2)

    return jobs


def find_remote_logs(hostname: str, niftytune_dir: str = '~/niftytune') -> list:
    """Find training log files on a remote machine."""
    r = _ssh_cmd(hostname,
                 f'ls -lt {niftytune_dir}/*.log 2>/dev/null | head -10',
                 timeout=10)
    logs = []
    seen = set()
    if r['ok']:
        for line in r['stdout'].strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 9:
                path = parts[-1]
                basename = os.path.basename(path)
                if basename in seen:
                    continue
                seen.add(basename)
                logs.append({
                    'path': path,
                    'name': basename,
                    'size': parts[4],
                    'modified': ' '.join(parts[5:8]),
                })
    return logs


def read_remote_log(hostname: str, log_path: str, tail_lines: int = 50) -> list:
    """Read the last N lines of a remote log file."""
    r = _ssh_cmd(hostname, f'tail -{tail_lines} {log_path}', timeout=10)
    if r['ok']:
        return r['stdout'].strip().split('\n')
    return []


# ---------------------------------------------------------------------------
# Remote job monitoring (streaming via SSH tail -f)
# ---------------------------------------------------------------------------

_remote_monitors = {}  # monitor_id -> {process, output_lines, metrics, ...}
_remote_lock = threading.Lock()

_ansi_re = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r')


def start_remote_monitor(hostname: str, log_path: str,
                         total_iters: int = 0,
                         job_id: str = None) -> str:
    """Start tailing a remote log file. Returns monitor_id."""
    monitor_id = f'remote_{int(time.time())}'

    cmd = ['ssh', '-o', 'ConnectTimeout=5', hostname,
           f'tail -n 100 -f {log_path}']

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        return None

    monitor = {
        'process': proc,
        'hostname': hostname,
        'log_path': log_path,
        'output_lines': [],
        'metrics': [],
        'done': False,
        'exit_code': None,
        'total_iters': total_iters,
        'best_val_loss': None,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'job_id': job_id,
    }

    with _remote_lock:
        _remote_monitors[monitor_id] = monitor

    t = threading.Thread(target=_read_remote_output,
                         args=(monitor_id,), daemon=True)
    t.start()

    return monitor_id


def _read_remote_output(monitor_id: str):
    """Background thread: read SSH tail output, parse metrics."""
    monitor = _remote_monitors.get(monitor_id)
    if not monitor:
        return

    proc = monitor['process']
    best_val = None

    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            clean = _ansi_re.sub('', line) if '\x1b' in line or '\r' in line else line
            if not clean.strip():
                continue

            with _remote_lock:
                monitor['output_lines'].append(clean)
                if len(monitor['output_lines']) > 10_000:
                    del monitor['output_lines'][:5_000]

            metric = log_parser.parse_line(clean)
            if metric:
                with _remote_lock:
                    monitor['metrics'].append(metric)
                if metric.get('type') == 'val' and metric.get('val_loss') is not None:
                    if best_val is None or metric['val_loss'] < best_val:
                        best_val = metric['val_loss']
                        monitor['best_val_loss'] = best_val

        proc.wait()
    except Exception:
        pass

    with _remote_lock:
        monitor['done'] = True
        monitor['exit_code'] = proc.returncode or 0

    # Update linked job in jobs.yaml
    if monitor.get('job_id'):
        _update_linked_job(monitor)

    # Evict after grace period
    def _evict():
        time.sleep(300)
        with _remote_lock:
            _remote_monitors.pop(monitor_id, None)
    threading.Thread(target=_evict, daemon=True).start()


def _update_linked_job(monitor: dict):
    """Update the linked job in jobs.yaml when remote training finishes."""
    try:
        import training
        job_id = monitor['job_id']

        # Get final train loss from metrics
        final_loss = None
        train_metrics = [m for m in monitor['metrics'] if m.get('type') == 'train']
        if train_metrics:
            final_loss = train_metrics[-1].get('train_loss')

        # Determine status — check if training completed normally
        # (SSH tail exits when the remote process ends, returncode may not reflect training status)
        status = 'completed'
        last_lines = monitor['output_lines'][-5:] if monitor['output_lines'] else []
        for line in last_lines:
            if 'error' in line.lower() or 'traceback' in line.lower():
                status = 'failed'
                break
            if 'Saved final weights' in line:
                status = 'completed'
                break

        training.update_job(job_id, {
            'status': status,
            'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
            'final_loss': final_loss,
            'best_val_loss': monitor.get('best_val_loss'),
        })
    except Exception:
        pass


def get_remote_monitor(monitor_id: str) -> dict | None:
    with _remote_lock:
        return _remote_monitors.get(monitor_id)


def stop_remote_monitor(monitor_id: str) -> bool:
    """Stop a remote monitor (kills the SSH tail process)."""
    with _remote_lock:
        monitor = _remote_monitors.get(monitor_id)
        if not monitor or monitor['done']:
            return False
        proc = monitor['process']

    try:
        proc.terminate()
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    with _remote_lock:
        monitor['done'] = True
    return True


def list_remote_monitors() -> list:
    with _remote_lock:
        return [
            {'id': mid, 'hostname': m['hostname'], 'log_path': m['log_path'],
             'done': m['done'], 'lines': len(m['output_lines']),
             'metrics': len(m['metrics']), 'started': m['started']}
            for mid, m in _remote_monitors.items()
        ]


# ---------------------------------------------------------------------------
# Remote training start/stop
# ---------------------------------------------------------------------------

def start_remote_training(hostname: str, niftytune_dir: str,
                          config_file: str, venv: str = 'venv_mlx',
                          session_name: str = 'orrvert') -> dict:
    """Start training on a remote machine via screen + SSH."""
    activate = f'source {niftytune_dir}/{venv}/bin/activate'
    train_cmd = f'python -u -m mlx_lm lora --config {config_file}'

    # Derive log path from config filename (just the basename)
    config_basename = os.path.basename(config_file)
    log_basename = config_basename.replace('.yaml', '.log').replace('.yml', '.log')
    if log_basename == config_basename:
        log_basename = 'training.log'
    log_path = f'{niftytune_dir}/{log_basename}'

    full_cmd = (
        f'cd {niftytune_dir} && {activate} && '
        f'screen -dmS {session_name} bash -c "'
        f'{train_cmd} 2>&1 | tee {log_path}"'
    )

    r = _ssh_cmd(hostname, full_cmd, timeout=15)
    if not r['ok']:
        return {'ok': False, 'error': r['stderr']}

    # Verify it started
    time.sleep(2)
    r2 = _ssh_cmd(hostname, f'screen -ls | grep {session_name}')
    started = r2['ok'] and session_name in r2['stdout']

    return {
        'ok': started,
        'log_path': log_path,
        'session_name': session_name,
    }


def stop_remote_training(hostname: str, session_name: str = 'orrvert',
                         pid: str = None) -> dict:
    """Stop training on a remote machine."""
    if pid:
        r = _ssh_cmd(hostname, f'kill {pid}')
    else:
        r = _ssh_cmd(hostname, f'screen -S {session_name} -X quit')

    # Also kill any mlx_lm processes
    _ssh_cmd(hostname, 'pkill -f "mlx_lm.*lora"', timeout=5)

    return {'ok': True, 'output': r['stdout'] + r['stderr']}


def list_remote_checkpoints(hostname: str, adapter_dir: str) -> list:
    """List adapter checkpoints on a remote machine."""
    r = _ssh_cmd(hostname,
                 f'ls -lt {adapter_dir}/*adapters*.safetensors 2>/dev/null',
                 timeout=10)
    checkpoints = []
    if r['ok']:
        for line in r['stdout'].strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 9:
                name = os.path.basename(parts[-1])
                checkpoints.append({
                    'name': name,
                    'path': parts[-1],
                    'size': parts[4],
                    'modified': ' '.join(parts[5:8]),
                })
    return checkpoints


def list_remote_adapters(hostname: str, niftytune_path: str = '~/niftytune') -> list:
    """List LoRA adapter directories on a remote machine.

    Looks for directories containing adapter markers (adapter_config.json,
    adapters.safetensors) under the niftytune path and common subdirectories.
    """
    # Find adapter directories by looking for marker files
    cmd = (
        f'find {niftytune_path} -maxdepth 3 '
        f'\\( -name adapter_config.json -o -name adapters.safetensors \\) '
        f'-printf "%h\\n" 2>/dev/null | sort -u'
    )
    # macOS uses -print instead of -printf, and stat differs
    r = _ssh_cmd(hostname, cmd, timeout=15)

    # Fallback for macOS (no -printf)
    if not r['ok'] or not r['stdout'].strip():
        cmd_mac = (
            f'find {niftytune_path} -maxdepth 3 '
            f'\\( -name adapter_config.json -o -name adapters.safetensors \\) '
            f'2>/dev/null | xargs -I{{}} dirname {{}} | sort -u'
        )
        r = _ssh_cmd(hostname, cmd_mac, timeout=15)

    if not r['ok'] or not r['stdout'].strip():
        return []

    adapters = []
    for adapter_dir in r['stdout'].strip().split('\n'):
        adapter_dir = adapter_dir.strip()
        if not adapter_dir:
            continue

        # Get size and file count
        stat_cmd = (
            f'du -sh "{adapter_dir}" 2>/dev/null && '
            f'ls -1 "{adapter_dir}"/*.safetensors 2>/dev/null | wc -l && '
            f'stat -f "%m" "{adapter_dir}" 2>/dev/null || '
            f'stat -c "%Y" "{adapter_dir}" 2>/dev/null'
        )
        stat_r = _ssh_cmd(hostname, stat_cmd, timeout=10)

        size = '?'
        checkpoint_count = 0
        mtime = ''
        if stat_r['ok']:
            lines = stat_r['stdout'].strip().split('\n')
            if lines:
                size = lines[0].split('\t')[0] if '\t' in lines[0] else lines[0].split()[0]
            if len(lines) >= 2:
                try:
                    checkpoint_count = int(lines[1].strip())
                except ValueError:
                    pass
            if len(lines) >= 3:
                try:
                    ts = int(lines[2].strip())
                    mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))
                except ValueError:
                    pass

        name = os.path.basename(adapter_dir)
        adapters.append({
            'path': adapter_dir,
            'name': name,
            'size': size,
            'checkpoints': checkpoint_count,
            'modified': mtime,
            'remote': True,
        })

    return sorted(adapters, key=lambda a: a.get('modified', ''), reverse=True)


# ---------------------------------------------------------------------------
# Adapter download (SCP)
# ---------------------------------------------------------------------------

_download_tasks = {}
_download_lock = threading.Lock()


def download_adapter(hostname: str, remote_path: str,
                     local_dir: str) -> str:
    """Download an adapter directory from a remote machine via SCP.

    Runs in background thread. Returns task_id for status polling.
    """
    task_id = f'dl_{int(time.time())}'
    adapter_name = os.path.basename(remote_path)
    local_dest = os.path.join(local_dir, adapter_name)

    task = {
        'id': task_id,
        'status': 'running',
        'hostname': hostname,
        'remote_path': remote_path,
        'local_path': local_dest,
        'adapter_name': adapter_name,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': None,
        'error': None,
    }
    with _download_lock:
        _download_tasks[task_id] = task

    def _run():
        try:
            os.makedirs(local_dir, exist_ok=True)
            proc = subprocess.run(
                ['scp', '-r', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                 f'{hostname}:{remote_path}', local_dest],
                capture_output=True, text=True,
                timeout=600,  # 10 min for large adapters
            )
            if proc.returncode == 0:
                task['status'] = 'completed'
            else:
                task['status'] = 'failed'
                task['error'] = proc.stderr.strip()
        except subprocess.TimeoutExpired:
            task['status'] = 'failed'
            task['error'] = 'SCP timed out (10 min limit)'
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
        task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Evict task after 5 minutes
        def _evict():
            time.sleep(300)
            with _download_lock:
                _download_tasks.pop(task_id, None)
        threading.Thread(target=_evict, daemon=True).start()

    threading.Thread(target=_run, daemon=True).start()
    return task_id


def get_download_task(task_id: str) -> dict | None:
    with _download_lock:
        return _download_tasks.get(task_id)


def list_download_tasks() -> list:
    with _download_lock:
        return list(_download_tasks.values())


# ---------------------------------------------------------------------------
# Wake-on-LAN
# ---------------------------------------------------------------------------

def send_wol(mac_address: str, broadcast: str = '255.255.255.255',
             port: int = 9) -> dict:
    """Send a Wake-on-LAN magic packet to wake a machine.

    mac_address: MAC in any format (aa:bb:cc:dd:ee:ff or aa-bb-cc-dd-ee-ff)
    """
    mac = mac_address.replace(':', '').replace('-', '').replace('.', '')
    if len(mac) != 12:
        return {'ok': False, 'error': f'Invalid MAC address: {mac_address}'}

    try:
        mac_bytes = bytes.fromhex(mac)
    except ValueError:
        return {'ok': False, 'error': f'Invalid MAC hex: {mac_address}'}

    # Magic packet: 6x 0xFF + 16x MAC address
    packet = b'\xff' * 6 + mac_bytes * 16

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(packet, (broadcast, port))
        sock.close()
        return {'ok': True, 'mac': mac_address}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def get_mac_address(hostname: str) -> str | None:
    """Try to get a machine's MAC address from ARP cache or SSH."""
    # Try ARP cache first (works if we've recently communicated)
    try:
        ip = hostname.split('@')[-1] if '@' in hostname else hostname
        r = subprocess.run(['arp', '-n', ip], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            # Parse: "192.168.1.132  ether  aa:bb:cc:dd:ee:ff  C  eth0"
            for line in r.stdout.split('\n'):
                parts = line.split()
                if len(parts) >= 3 and ':' in parts[2]:
                    return parts[2]
        # Try ip neigh as fallback
        r = subprocess.run(['ip', 'neigh', 'show', ip],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            for line in r.stdout.split('\n'):
                parts = line.split()
                if 'lladdr' in parts:
                    idx = parts.index('lladdr') + 1
                    if idx < len(parts):
                        return parts[idx]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try SSH as fallback (machine must be on)
    r = _ssh_cmd(hostname,
                 "ifconfig 2>/dev/null | grep -o 'ether [a-f0-9:]*' | head -1 | cut -d' ' -f2 || "
                 "ip link show 2>/dev/null | grep -o 'link/ether [a-f0-9:]*' | head -1 | cut -d' ' -f2",
                 timeout=10)
    if r['ok'] and r['stdout'].strip():
        return r['stdout'].strip()

    return None


# ---------------------------------------------------------------------------
# Job queue for offline machines
# ---------------------------------------------------------------------------

_job_queue = []  # list of {id, machine, action, params, queued_at}
_queue_lock = threading.Lock()
_queue_watcher = None  # background thread


def queue_job(machine: str, action: str, params: dict) -> str:
    """Queue a job for an offline machine. Returns queue entry ID."""
    entry_id = f'q_{int(time.time())}_{len(_job_queue)}'
    entry = {
        'id': entry_id,
        'machine': machine,
        'action': action,
        'params': params,
        'status': 'queued',
        'queued_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'started_at': None,
        'error': None,
    }
    with _queue_lock:
        _job_queue.append(entry)
    _ensure_queue_watcher()
    return entry_id


def list_queue() -> list:
    with _queue_lock:
        return [dict(e) for e in _job_queue]


def cancel_queued(entry_id: str) -> bool:
    with _queue_lock:
        for i, e in enumerate(_job_queue):
            if e['id'] == entry_id and e['status'] == 'queued':
                _job_queue.pop(i)
                return True
    return False


def _ensure_queue_watcher():
    """Start the background queue watcher if not running."""
    global _queue_watcher
    if _queue_watcher and _queue_watcher.is_alive():
        return
    _queue_watcher = threading.Thread(target=_watch_queue, daemon=True)
    _queue_watcher.start()


def _watch_queue():
    """Poll queued jobs, dispatch when machines come online."""
    while True:
        with _queue_lock:
            pending = [e for e in _job_queue if e['status'] == 'queued']
        if not pending:
            time.sleep(30)
            continue

        # Group by machine, test each once
        machines_tested = {}
        for entry in pending:
            machine = entry['machine']
            if machine not in machines_tested:
                # Load machine config to get hostname
                try:
                    config_path = os.path.join(CONFIG_DIR, 'machines.yaml')
                    with open(config_path) as f:
                        data = yaml.safe_load(f) or {}
                    m = data.get('machines', {}).get(machine, {})
                    hostname = m.get('hostname', '')
                    if hostname:
                        result = test_connection(hostname)
                        machines_tested[machine] = result.get('reachable', False)
                    else:
                        machines_tested[machine] = False
                except Exception:
                    machines_tested[machine] = False

            if machines_tested.get(machine):
                # Machine is online — dispatch the job
                entry['status'] = 'dispatching'
                entry['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                _dispatch_queued(entry)

        # Clean up completed/failed entries older than 1 hour
        cutoff = time.time() - 3600
        with _queue_lock:
            _job_queue[:] = [
                e for e in _job_queue
                if e['status'] == 'queued' or
                (e.get('started_at') and
                 time.mktime(time.strptime(e['started_at'], '%Y-%m-%d %H:%M:%S')) > cutoff)
            ]

        time.sleep(60)  # Check every minute


def _dispatch_queued(entry: dict):
    """Dispatch a queued job to its target machine."""
    try:
        action = entry['action']
        params = entry['params']

        if action == 'wol':
            # Just a wake — already online, mark done
            entry['status'] = 'completed'
        elif action == 'train':
            # Import here to avoid circular dependency
            # The actual dispatch is handled by the caller via callback
            entry['status'] = 'dispatched'
        else:
            entry['status'] = 'dispatched'
    except Exception as e:
        entry['status'] = 'failed'
        entry['error'] = str(e)
