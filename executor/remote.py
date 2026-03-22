"""Remote machine dispatch for orracle-dev pipeline jobs.

Reads machine definitions from config/machines.yaml.
Launches run_pipeline.py on a remote machine inside a screen session.
Streams logs back via SSH tail -f in a background thread.
SCPs train.jsonl + val.jsonl back on demand.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import uuid

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
MACHINES_CONFIG = os.path.join(CONFIG_DIR, 'machines.yaml')
SCREEN_PREFIX = 'orracle'
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r')


# ---------------------------------------------------------------------------
# Machine registry
# ---------------------------------------------------------------------------

def load_machines() -> dict:
    if not os.path.exists(MACHINES_CONFIG):
        return {}
    with open(MACHINES_CONFIG) as f:
        data = yaml.safe_load(f) or {}
    return data.get('machines', {})


def get_machine(name: str) -> dict | None:
    return load_machines().get(name)


def _ssh_target(machine: dict) -> str:
    return f"{machine['user']}@{machine['host']}"


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh(target: str, command: str, timeout: int = 10) -> dict:
    try:
        # Force bash to avoid fish/zsh shell issues on remote machines
        r = subprocess.run(
            ['ssh',
             '-o', 'ConnectTimeout=5',
             '-o', 'BatchMode=yes',
             '-o', 'StrictHostKeyChecking=accept-new',
             target, f'bash -c {repr(command)}'],
            capture_output=True, text=True, timeout=timeout,
        )
        # Filter out remote shell startup noise (fish fastfetch, etc.)
        stderr = '\n'.join(
            line for line in r.stderr.splitlines()
            if not any(x in line for x in
                       ['fastfetch', 'config.fish', 'from sourcing',
                        'called during', 'Unknown command', '^~'])
        ).strip()
        return {
            'ok': r.returncode == 0,
            'stdout': r.stdout,
            'stderr': stderr,
            'returncode': r.returncode,
        }
    except subprocess.TimeoutExpired:
        return {'ok': False, 'stdout': '', 'stderr': 'SSH timeout', 'returncode': -1}
    except FileNotFoundError:
        return {'ok': False, 'stdout': '', 'stderr': 'ssh not found', 'returncode': -1}


def test_connection(machine_name: str) -> dict:
    machine = get_machine(machine_name)
    if not machine:
        return {'reachable': False, 'error': f'Unknown machine: {machine_name}'}
    r = _ssh(_ssh_target(machine), 'echo ok', timeout=8)
    reachable = 'ok' in r.get('stdout', '')
    return {
        'reachable': reachable,
        'error': r['stderr'].strip() if not reachable else None,
    }


def check_vault(machine_name: str) -> dict:
    machine = get_machine(machine_name)
    if not machine:
        return {'ok': False, 'error': f'Unknown machine: {machine_name}'}
    vault = machine.get('vault_path', '')
    if not vault:
        return {'ok': False, 'error': 'No vault_path configured'}
    r = _ssh(_ssh_target(machine),
             f'ls "{vault}" >/dev/null 2>&1 && echo __ok__',
             timeout=15)
    if '__ok__' in r.get('stdout', ''):
        return {'ok': True, 'vault_path': vault}
    return {'ok': False, 'error': r['stderr'].strip() or 'Vault path not accessible'}


# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(machine_name: str, env_overrides: dict | None = None) -> str:
    job_id = f'remote_{uuid.uuid4().hex[:8]}'
    machine = get_machine(machine_name)
    session_name = f'{SCREEN_PREFIX}-{job_id[7:]}'

    job: dict = {
        'id': job_id,
        'machine': machine_name,
        'status': 'dispatching',
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': None,
        'log': [],
        'error': None,
        'screen_session': session_name,
        'log_path': None,
        'target': _ssh_target(machine) if machine else None,
    }

    with _jobs_lock:
        _jobs[job_id] = job

    threading.Thread(
        target=_dispatch_thread,
        args=(job_id, machine, env_overrides),
        daemon=True,
    ).start()

    return job_id


def _dispatch_thread(job_id: str, machine: dict | None, env_overrides: dict | None):
    job = _jobs[job_id]

    if not machine:
        job['status'] = 'failed'
        job['error'] = 'Machine not found in config'
        job['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        return

    target = _ssh_target(machine)
    project_path = machine.get('project_path', '~/projects/orracle-dev')
    vault_path = machine.get('vault_path', '')
    session = job['screen_session']

    log_path = f'{project_path}/output/{job_id}.log'
    job['log_path'] = log_path

    env_parts: list[str] = []
    if vault_path:
        env_parts.append(f'ORRACLE_SOURCE="{vault_path}"')
    if env_overrides:
        for k, v in env_overrides.items():
            env_parts.append(f'{k}="{v}"')
    env_str = (' '.join(env_parts) + ' ') if env_parts else ''

    venv_python = f'{project_path}/venv/bin/python'
    python_cmd = f'$([ -f {venv_python} ] && echo {venv_python} || echo python3)'

    run_cmd = (
        f'cd {project_path} && '
        f'mkdir -p {project_path}/output && '
        f'{env_str}{python_cmd} run_pipeline.py'
    )

    screen_cmd = (
        f'screen -dmS {session} bash -c '
        f'"{run_cmd} 2>&1 | tee {log_path}; '
        f'echo __DONE_$?__ >> {log_path}"'
    )

    job['log'].append(f'Dispatching to {target}')
    job['log'].append(f'Session: {session}')
    r = _ssh(target, screen_cmd, timeout=20)

    if not r['ok']:
        job['status'] = 'failed'
        job['error'] = r['stderr'].strip() or 'Failed to launch screen session'
        job['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        return

    time.sleep(2)
    r2 = _ssh(target, f'screen -ls 2>/dev/null | grep {session}', timeout=8)
    if not (r2['ok'] and session in r2.get('stdout', '')):
        job['status'] = 'failed'
        job['error'] = 'Screen session did not appear after launch'
        job['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        return

    job['status'] = 'running'
    job['log'].append('Screen session confirmed running')
    job['log'].append(f'Log path: {log_path}')

    _start_log_monitor(job_id, target, log_path)


# ---------------------------------------------------------------------------
# Log streaming
# ---------------------------------------------------------------------------

def _start_log_monitor(job_id: str, target: str, log_path: str):
    def _monitor():
        job = _jobs.get(job_id)
        if not job:
            return

        for _ in range(30):
            r = _ssh(target, f'test -f {log_path} && echo yes', timeout=8)
            if r['ok'] and 'yes' in r['stdout']:
                break
            time.sleep(2)
        else:
            with _jobs_lock:
                job['log'].append('WARNING: Log file did not appear within 60s')

        cmd = [
            'ssh',
            '-o', 'ConnectTimeout=5',
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ServerAliveInterval=30',
            target,
            f'tail -n 0 -f {log_path}',
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except Exception as e:
            with _jobs_lock:
                job['log'].append(f'ERROR: Could not start log tail: {e}')
                job['status'] = 'failed'
                job['error'] = str(e)
                job['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
            return

        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip('\n')
                if not line.strip():
                    continue
                clean = _ANSI_RE.sub('', line) if '\x1b' in line else line
                if not clean.strip():
                    continue

                with _jobs_lock:
                    job['log'].append(clean)
                    if len(job['log']) > 10_000:
                        del job['log'][:5_000]

                if clean.startswith('__DONE_'):
                    rc_str = clean[len('__DONE_'):].strip('_')
                    try:
                        rc = int(rc_str)
                    except ValueError:
                        rc = -1
                    with _jobs_lock:
                        job['status'] = 'completed' if rc == 0 else 'failed'
                        if rc != 0:
                            job['error'] = f'Pipeline exited with code {rc}'
                        job['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    proc.terminate()
                    break

            proc.wait()
        except Exception as e:
            with _jobs_lock:
                job['log'].append(f'Log monitor error: {e}')

        with _jobs_lock:
            if job.get('status') == 'running':
                job['status'] = 'unknown'
                job['log'].append(
                    'SSH tail exited without completion marker — '
                    'check remote machine')
                job['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')

    threading.Thread(target=_monitor, daemon=True).start()


# ---------------------------------------------------------------------------
# Job queries
# ---------------------------------------------------------------------------

def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


def list_jobs() -> list[dict]:
    with _jobs_lock:
        return sorted(
            [{
                'id': j['id'], 'machine': j['machine'],
                'status': j['status'], 'started': j['started'],
                'finished': j['finished'], 'log_lines': len(j['log']),
                'screen_session': j.get('screen_session'),
            } for j in _jobs.values()],
            key=lambda j: j['started'], reverse=True,
        )


def check_remote_status(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        return {'error': 'Job not found'}
    machine = get_machine(job['machine'])
    if not machine:
        return {'error': 'Machine config not found'}
    session = job.get('screen_session', '')
    r = _ssh(_ssh_target(machine),
             f'screen -ls 2>/dev/null | grep {session}', timeout=8)
    return {
        'job_id': job_id, 'session': session,
        'running': r['ok'] and session in r.get('stdout', ''),
    }


def stop_job(job_id: str) -> dict:
    job = get_job(job_id)
    if not job:
        return {'ok': False, 'error': 'Job not found'}
    machine = get_machine(job['machine'])
    if not machine:
        return {'ok': False, 'error': 'Machine config not found'}
    session = job.get('screen_session', '')
    r = _ssh(_ssh_target(machine),
             f'screen -S {session} -X quit 2>/dev/null; echo done', timeout=10)
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]['status'] = 'cancelled'
            _jobs[job_id]['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    return {'ok': r['ok'], 'session': session}


# ---------------------------------------------------------------------------
# Output retrieval
# ---------------------------------------------------------------------------

_downloads: dict[str, dict] = {}
_downloads_lock = threading.Lock()


def fetch_output(job_id: str, local_output_dir: str) -> str:
    dl_id = f'dl_{uuid.uuid4().hex[:8]}'
    job = get_job(job_id)

    task: dict = {
        'id': dl_id, 'job_id': job_id, 'status': 'running',
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': None, 'files': [], 'error': None,
    }
    with _downloads_lock:
        _downloads[dl_id] = task

    def _run():
        if not job:
            task['status'] = 'failed'
            task['error'] = 'Job not found'
            task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
            return

        machine = get_machine(job['machine'])
        if not machine:
            task['status'] = 'failed'
            task['error'] = 'Machine config not found'
            task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
            return

        target = _ssh_target(machine)
        project_path = machine.get('project_path', '~/projects/orracle-dev')
        remote_output = f'{project_path}/output'
        os.makedirs(local_output_dir, exist_ok=True)

        files_fetched = []
        errors = []

        for fname in ('train.jsonl', 'val.jsonl'):
            remote_src = f'{target}:{remote_output}/{fname}'
            local_dest = os.path.join(local_output_dir, fname)
            try:
                r = subprocess.run(
                    ['scp', '-o', 'ConnectTimeout=10',
                     '-o', 'BatchMode=yes', remote_src, local_dest],
                    capture_output=True, text=True, timeout=300,
                )
                if r.returncode == 0:
                    size = os.path.getsize(local_dest)
                    files_fetched.append({
                        'name': fname, 'local': local_dest,
                        'size_mb': round(size / (1024 ** 2), 2),
                    })
                else:
                    errors.append(f'{fname}: {r.stderr.strip()}')
            except subprocess.TimeoutExpired:
                errors.append(f'{fname}: SCP timed out')
            except Exception as e:
                errors.append(f'{fname}: {e}')

        task['files'] = files_fetched
        task['status'] = 'completed' if files_fetched else 'failed'
        if errors:
            task['error'] = '; '.join(errors)
        task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')

    threading.Thread(target=_run, daemon=True).start()
    return dl_id


def get_download(dl_id: str) -> dict | None:
    with _downloads_lock:
        d = _downloads.get(dl_id)
        return dict(d) if d else None
