from __future__ import annotations

"""Model export and Ollama management.

Handles: Ollama model lifecycle, adapter scanning, GGUF file management,
Modelfile generation, background export tasks, and full deploy pipelines
(remote fuse → transfer → ollama create).
"""

import os
import re
import subprocess
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Scan paths
# ---------------------------------------------------------------------------

NIFTYTUNE_DIR = Path.home() / 'projects' / 'orrapus' / 'niftytune'
ORRACLE_DIR = Path(__file__).parent

DEFAULT_ADAPTER_DIRS = [
    NIFTYTUNE_DIR / 'models' / 'checkpoints',
    NIFTYTUNE_DIR / 'adapters',
    ORRACLE_DIR / 'config',  # local adapters from training jobs
]

DEFAULT_GGUF_DIRS = [
    NIFTYTUNE_DIR / 'models' / 'gguf',
    Path.home() / '.ollama' / 'models',
]


# ---------------------------------------------------------------------------
# Ollama management
# ---------------------------------------------------------------------------

def list_ollama_models() -> list:
    """List all Ollama models via `ollama list`."""
    try:
        r = subprocess.run(['ollama', 'list'],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return []

        models = []
        lines = r.stdout.strip().split('\n')
        if len(lines) < 2:
            return []

        for line in lines[1:]:  # skip header
            # Format: NAME  ID  SIZE  MODIFIED
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 4:
                models.append({
                    'name': parts[0],
                    'id': parts[1],
                    'size': parts[2],
                    'modified': parts[3],
                })
            elif len(parts) >= 3:
                models.append({
                    'name': parts[0],
                    'id': parts[1],
                    'size': parts[2],
                    'modified': '',
                })
        return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def running_ollama_models() -> list:
    """List currently loaded/running Ollama models via `ollama ps`."""
    try:
        r = subprocess.run(['ollama', 'ps'],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return []
        lines = r.stdout.strip().split('\n')
        if len(lines) < 2:
            return []
        models = []
        for line in lines[1:]:
            parts = re.split(r'\s{2,}', line.strip())
            if parts:
                models.append({'name': parts[0]})
        return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def show_ollama_modelfile(name: str) -> str | None:
    """Get the Modelfile for an Ollama model."""
    try:
        r = subprocess.run(['ollama', 'show', '--modelfile', name],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def create_ollama_model(name: str, modelfile_content: str) -> dict:
    """Create an Ollama model from a Modelfile string."""
    tmp_path = Path('/tmp') / f'orracle_modelfile_{int(time.time())}'
    try:
        tmp_path.write_text(modelfile_content)
        r = subprocess.run(
            ['ollama', 'create', name, '-f', str(tmp_path)],
            capture_output=True, text=True, timeout=600,
        )
        return {
            'ok': r.returncode == 0,
            'stdout': r.stdout,
            'stderr': r.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {'ok': False, 'stderr': str(e)}
    finally:
        tmp_path.unlink(missing_ok=True)


def delete_ollama_model(name: str) -> dict:
    """Delete an Ollama model."""
    try:
        r = subprocess.run(['ollama', 'rm', name],
                           capture_output=True, text=True, timeout=30)
        return {'ok': r.returncode == 0, 'stdout': r.stdout, 'stderr': r.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {'ok': False, 'stderr': str(e)}


def copy_ollama_model(source: str, dest: str) -> dict:
    """Copy/alias an Ollama model."""
    try:
        r = subprocess.run(['ollama', 'cp', source, dest],
                           capture_output=True, text=True, timeout=30)
        return {'ok': r.returncode == 0, 'stdout': r.stdout, 'stderr': r.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {'ok': False, 'stderr': str(e)}


# ---------------------------------------------------------------------------
# Adapter & GGUF scanning
# ---------------------------------------------------------------------------

def list_adapters(extra_dirs: list = None) -> list:
    """Scan for LoRA adapter checkpoint directories."""
    dirs = [d for d in DEFAULT_ADAPTER_DIRS if d.exists()]
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs if Path(d).exists())

    adapters = []
    seen = set()
    for scan_dir in dirs:
        for root, _, filenames in os.walk(str(scan_dir), followlinks=False):
            root_path = Path(root)
            # LoRA markers: adapter_config.json (HF/Unsloth) or adapters.safetensors (mlx-lm)
            has_marker = ('adapter_config.json' in filenames or
                          'adapters.safetensors' in filenames or
                          'adapter_model.safetensors' in filenames)
            if not has_marker:
                continue
            real = str(root_path.resolve())
            if real in seen:
                continue
            seen.add(real)

            total_size = sum(
                f.stat().st_size for f in root_path.iterdir()
                if f.is_file()
            )
            adapters.append({
                'path': str(root_path),
                'name': root_path.name,
                'parent': root_path.parent.name,
                'size_mb': round(total_size / (1024 * 1024), 1),
                'modified': time.strftime(
                    '%Y-%m-%d %H:%M',
                    time.localtime(root_path.stat().st_mtime)
                ),
            })

    return sorted(adapters, key=lambda a: a.get('modified', ''), reverse=True)


def list_gguf_files(extra_dirs: list = None) -> list:
    """Scan for GGUF model files."""
    dirs = [d for d in DEFAULT_GGUF_DIRS if d.exists()]
    if extra_dirs:
        dirs.extend(Path(d) for d in extra_dirs if Path(d).exists())

    files = []
    seen = set()
    for scan_dir in dirs:
        for f in scan_dir.rglob('*.gguf'):
            real = str(f.resolve())
            if real in seen:
                continue
            seen.add(real)
            stat = f.stat()
            files.append({
                'path': str(f),
                'name': f.name,
                'dir': str(f.parent),
                'size_mb': round(stat.st_size / (1024 * 1024), 1),
                'size_gb': round(stat.st_size / (1024 ** 3), 2),
                'modified': time.strftime(
                    '%Y-%m-%d %H:%M',
                    time.localtime(stat.st_mtime)
                ),
            })

    return sorted(files, key=lambda x: x.get('modified', ''), reverse=True)


# ---------------------------------------------------------------------------
# Modelfile generation
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a creative fiction writer specializing in character-driven stories. "
    "Write vivid, emotionally resonant prose with natural dialogue and authentic "
    "character development."
)

MISTRAL_TEMPLATE = (
    '{{ if .System }}<s>[INST] {{ .System }}\n\n'
    '{{ .Prompt }} [/INST]{{ else }}<s>[INST] {{ .Prompt }} [/INST]{{ end }}'
    '{{ .Response }}</s>'
)

LLAMA3_TEMPLATE = (
    '<|begin_of_text|>{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n'
    '{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>\n\n'
    '{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    '{{ .Response }}<|eot_id|>'
)

CHAT_TEMPLATES = {
    'mistral': {'name': 'Mistral', 'template': MISTRAL_TEMPLATE,
                'stop': ['</s>', '[INST]']},
    'llama3': {'name': 'Llama 3', 'template': LLAMA3_TEMPLATE,
               'stop': ['<|eot_id|>']},
    'chatml': {'name': 'ChatML (Qwen)', 'template':
               '<|im_start|>system\n{{ .System }}<|im_end|>\n'
               '<|im_start|>user\n{{ .Prompt }}<|im_end|>\n'
               '<|im_start|>assistant\n{{ .Response }}<|im_end|>',
               'stop': ['<|im_end|>']},
}


def generate_modelfile(gguf_path: str, system_prompt: str = None,
                       template_key: str = 'mistral',
                       params: dict = None) -> str:
    """Generate Ollama Modelfile content."""
    lines = [f'FROM {gguf_path}']

    tmpl = CHAT_TEMPLATES.get(template_key, CHAT_TEMPLATES['mistral'])
    lines.append(f'\nTEMPLATE """{tmpl["template"]}"""')

    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    lines.append(f'\nSYSTEM """{sys_prompt}"""')

    default_params = {
        'temperature': 0.8,
        'top_p': 0.92,
        'top_k': 50,
        'repeat_penalty': 1.1,
        'num_ctx': 2048,
    }
    if params:
        default_params.update(params)

    lines.append('')
    for key, value in default_params.items():
        lines.append(f'PARAMETER {key} {value}')

    for stop_token in tmpl.get('stop', []):
        lines.append(f'PARAMETER stop "{stop_token}"')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Background export tasks
# ---------------------------------------------------------------------------

_export_tasks = {}
_export_lock = threading.Lock()


def start_fuse_task(adapter_path: str, base_model: str,
                    output_dir: str, framework: str = 'mlx') -> str:
    """Start a LoRA fusion task in the background. Returns task_id."""
    task_id = f'fuse_{int(time.time())}'

    if framework == 'mlx':
        cmd = ['python', '-m', 'mlx_lm.fuse',
               '--model', base_model,
               '--adapter-path', adapter_path,
               '--save-path', output_dir]
    else:
        # Unsloth merge
        script = (
            f'from unsloth import FastLanguageModel; '
            f'm,t = FastLanguageModel.from_pretrained("{adapter_path}", '
            f'max_seq_length=2048, dtype=None, load_in_4bit=True); '
            f'm.save_pretrained_merged("{output_dir}", t, save_method="merged_16bit"); '
            f'print("Merge complete")'
        )
        cmd = ['python', '-c', script]

    return _run_export_task(task_id, 'fuse', cmd, cwd=str(NIFTYTUNE_DIR))


def start_gguf_task(adapter_path: str, output_name: str,
                    quant_type: str = 'q4_k_m',
                    framework: str = 'unsloth') -> str:
    """Start a GGUF export task. Returns task_id."""
    task_id = f'gguf_{int(time.time())}'
    output_dir = str(NIFTYTUNE_DIR / 'models' / 'gguf')

    if framework == 'unsloth':
        script = (
            f'from unsloth import FastLanguageModel; '
            f'm,t = FastLanguageModel.from_pretrained("{adapter_path}", '
            f'max_seq_length=2048, dtype=None, load_in_4bit=True); '
            f'm.save_pretrained_gguf("{output_dir}/{output_name}", t, '
            f'quantization_method="{quant_type}"); '
            f'print("GGUF export complete")'
        )
        cmd = ['python', '-c', script]
    else:
        cmd = ['python', '-m', 'mlx_lm.convert',
               '--hf-path', adapter_path,
               '--mlx-path', output_dir, '-q']

    return _run_export_task(task_id, 'gguf', cmd, cwd=str(NIFTYTUNE_DIR))


def _run_export_task(task_id: str, task_type: str,
                     cmd: list, cwd: str = None) -> str:
    """Run a background export task."""
    task = {
        'id': task_id,
        'type': task_type,
        'status': 'running',
        'command': cmd,
        'output': [],
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': None,
        'exit_code': None,
    }
    with _export_lock:
        _export_tasks[task_id] = task

    def _run():
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=cwd,
            )
            for line in proc.stdout:
                with _export_lock:
                    task['output'].append(line.rstrip())
                    if len(task['output']) > 2_000:
                        del task['output'][:1_000]
            proc.wait()
            task['status'] = 'completed' if proc.returncode == 0 else 'failed'
            task['exit_code'] = proc.returncode
        except Exception as e:
            task['status'] = 'failed'
            task['output'].append(f'Error: {e}')
        task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Evict after grace period
        def _evict():
            time.sleep(300)
            with _export_lock:
                _export_tasks.pop(task_id, None)
        threading.Thread(target=_evict, daemon=True).start()

    threading.Thread(target=_run, daemon=True).start()
    return task_id


def get_export_task(task_id: str) -> dict | None:
    with _export_lock:
        t = _export_tasks.get(task_id)
        if t:
            return {**t, 'output': list(t['output'])}
        return None


def list_export_tasks() -> list:
    with _export_lock:
        return sorted(
            [{'id': t['id'], 'type': t['type'], 'status': t['status'],
              'started': t['started'], 'finished': t['finished'],
              'stage': t.get('stage', '')}
             for t in _export_tasks.values()],
            key=lambda t: t['started'], reverse=True
        )


# ---------------------------------------------------------------------------
# Full deploy pipeline: remote fuse → SCP → GGUF convert → ollama create
# ---------------------------------------------------------------------------

LOCAL_MODEL_DIR = Path.home() / 'models'
CONVERT_VENV = Path(__file__).parent / 'venv_convert'
CONVERT_SCRIPT = Path(__file__).parent / 'scripts' / 'convert_hf_to_gguf.py'


def start_deploy_pipeline(
    adapter_path: str,
    base_model: str,
    model_name: str,
    hostname: str = None,
    niftytune_dir: str = '~/niftytune',
    venv: str = 'venv_mlx',
    framework: str = 'mlx',
    template_key: str = 'mistral',
    system_prompt: str = None,
    params: dict = None,
) -> str:
    """Full pipeline: fuse → GGUF convert → ollama create. Returns task_id.

    MLX path (remote): SSH fuse --dequantize → SCP safetensors → local
    convert_hf_to_gguf.py → ollama create.

    Unsloth path: fuse + GGUF in one step via save_pretrained_gguf.
    """
    task_id = f'deploy_{int(time.time())}'

    task = {
        'id': task_id,
        'type': 'deploy',
        'status': 'running',
        'stage': 'starting',
        'model_name': model_name,
        'command': [],
        'output': [],
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': None,
        'exit_code': None,
    }
    with _export_lock:
        _export_tasks[task_id] = task

    def _log(msg):
        with _export_lock:
            task['output'].append(msg)
            if len(task['output']) > 2_000:
                del task['output'][:1_000]

    def _set_stage(stage):
        with _export_lock:
            task['stage'] = stage

    def _fail(msg):
        _log(f'ERROR: {msg}')
        task['status'] = 'failed'
        task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')

    def _run():
        remote = hostname is not None
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        local_gguf = str(LOCAL_MODEL_DIR / f'{model_name}.gguf')
        local_fused = str(LOCAL_MODEL_DIR / f'fused_{model_name}')

        # ------- Stage 1: Fuse adapter -------
        _set_stage('fusing')

        if framework == 'unsloth':
            # Unsloth can fuse + GGUF in one step
            _log('[1/3] Fusing adapter + GGUF export (Unsloth)...')
            fuse_cmd = (
                f'from unsloth import FastLanguageModel; '
                f'm,t = FastLanguageModel.from_pretrained("{adapter_path}", '
                f'max_seq_length=2048, dtype=None, load_in_4bit=True); '
                f'm.save_pretrained_gguf("{local_gguf.replace(".gguf", "")}", t, '
                f'quantization_method="q4_k_m"); '
                f'print("GGUF export complete")'
            )
            if remote:
                r = subprocess.run(
                    ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                     hostname, f'cd {niftytune_dir} && {venv}/bin/python -c \'{fuse_cmd}\''],
                    capture_output=True, text=True, timeout=1800,
                )
                if r.returncode != 0:
                    _fail(f'Remote Unsloth fuse failed: {r.stderr.strip()}')
                    return
                for line in r.stdout.strip().split('\n'):
                    _log(line)
            else:
                try:
                    proc = subprocess.Popen(
                        ['python', '-c', fuse_cmd],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    for line in proc.stdout:
                        _log(line.rstrip())
                    proc.wait()
                    if proc.returncode != 0:
                        _fail(f'Unsloth fuse failed (exit {proc.returncode})')
                        return
                except Exception as e:
                    _fail(f'Fuse error: {e}')
                    return

            _set_stage('transferring')
            _log('[2/3] Skipped — Unsloth produced GGUF directly.')

        elif remote:
            # MLX remote: fuse --dequantize → SCP safetensors → local GGUF convert
            _log(f'[1/4] Fusing adapter on {hostname} (dequantize to f16)...')
            remote_fused = f'{niftytune_dir}/fused_{model_name}'

            fuse_cmd = (
                f'cd {niftytune_dir} && '
                f'{venv}/bin/python -m mlx_lm.fuse '
                f'--model {base_model} '
                f'--adapter-path {adapter_path} '
                f'--save-path {remote_fused} '
                f'--dequantize'
            )

            try:
                proc = subprocess.run(
                    ['ssh', '-o', 'ConnectTimeout=10', '-o', 'BatchMode=yes',
                     hostname, fuse_cmd],
                    capture_output=True, text=True,
                    timeout=1800,
                )
                for line in proc.stdout.strip().split('\n'):
                    if line:
                        _log(line)
                if proc.returncode != 0:
                    for line in proc.stderr.strip().split('\n'):
                        if line:
                            _log(f'stderr: {line}')
                    _fail(f'Remote fuse failed (exit {proc.returncode})')
                    return
                _log('Fuse complete (f16 safetensors).')
            except subprocess.TimeoutExpired:
                _fail('Remote fuse timed out (30 min)')
                return
            except Exception as e:
                _fail(f'SSH error: {e}')
                return

            # SCP fused safetensors back
            _set_stage('transferring')
            _log(f'[2/4] Transferring fused model from {hostname}...')

            try:
                proc = subprocess.run(
                    ['scp', '-r', '-o', 'ConnectTimeout=10',
                     '-o', 'BatchMode=yes',
                     f'{hostname}:{remote_fused}', local_fused],
                    capture_output=True, text=True,
                    timeout=1200,  # 20 min for ~14GB
                )
                if proc.returncode != 0:
                    _fail(f'SCP failed: {proc.stderr.strip()}')
                    return
                _log('Transfer complete.')
            except subprocess.TimeoutExpired:
                _fail('SCP timed out (20 min)')
                return
            except Exception as e:
                _fail(f'SCP error: {e}')
                return

            # Convert to GGUF locally
            _set_stage('converting')
            _log('[3/4] Converting to GGUF (Q8_0)...')

            convert_python = str(CONVERT_VENV / 'bin' / 'python')
            convert_script = str(CONVERT_SCRIPT)

            if not os.path.exists(convert_python):
                _fail(f'Conversion venv not found at {CONVERT_VENV}. '
                      f'Run: python3.12 -m venv venv_convert && '
                      f'venv_convert/bin/pip install torch transformers '
                      f'safetensors gguf numpy sentencepiece')
                return

            try:
                proc = subprocess.Popen(
                    [convert_python, convert_script, local_fused,
                     '--outfile', local_gguf, '--outtype', 'q8_0'],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    stripped = line.rstrip()
                    if stripped:
                        _log(stripped)
                proc.wait()
                if proc.returncode != 0:
                    _fail(f'GGUF conversion failed (exit {proc.returncode})')
                    return
                size_mb = os.path.getsize(local_gguf) / (1024 * 1024)
                _log(f'GGUF conversion complete ({size_mb:.0f} MB)')
            except Exception as e:
                _fail(f'Conversion error: {e}')
                return

        else:
            # MLX local fuse (less common — would need mlx on this machine)
            _log('[1/3] Fusing adapter locally (dequantize)...')
            cmd = [
                'python', '-m', 'mlx_lm.fuse',
                '--model', base_model,
                '--adapter-path', adapter_path,
                '--save-path', local_fused,
                '--dequantize',
            ]
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    _log(line.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    _fail(f'Local fuse failed (exit {proc.returncode})')
                    return
            except Exception as e:
                _fail(f'Fuse error: {e}')
                return

            _set_stage('converting')
            _log('[2/3] Converting to GGUF (Q8_0)...')
            convert_python = str(CONVERT_VENV / 'bin' / 'python')
            try:
                proc = subprocess.Popen(
                    [convert_python, str(CONVERT_SCRIPT), local_fused,
                     '--outfile', local_gguf, '--outtype', 'q8_0'],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in proc.stdout:
                    _log(line.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    _fail(f'GGUF conversion failed (exit {proc.returncode})')
                    return
            except Exception as e:
                _fail(f'Conversion error: {e}')
                return

        # ------- Final stage: ollama create -------
        _set_stage('creating')
        step = '4/4' if (remote and framework == 'mlx') else '3/3'
        _log(f'[{step}] Creating Ollama model "{model_name}"...')

        modelfile = generate_modelfile(
            local_gguf, system_prompt=system_prompt,
            template_key=template_key, params=params,
        )
        _log(f'Modelfile:\n{modelfile}')

        result = create_ollama_model(model_name, modelfile)
        if result.get('ok'):
            _log(f'Ollama model "{model_name}" created successfully!')
            task['status'] = 'completed'
            task['exit_code'] = 0
        else:
            stderr = result.get('stderr', 'unknown error')
            _fail(f'ollama create failed: {stderr}')
            return

        task['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        _set_stage('done')

        # Evict after grace period
        def _evict():
            time.sleep(600)
            with _export_lock:
                _export_tasks.pop(task_id, None)
        threading.Thread(target=_evict, daemon=True).start()

    threading.Thread(target=_run, daemon=True).start()
    return task_id
