"""ComfyUI API client and workflow builder.

Connects to ComfyUI's REST API to:
- List available checkpoints, LoRAs, samplers, schedulers
- Build and submit generation workflows
- Build LoRA bake workflows (merge LoRA into checkpoint)
- Poll for prompt completion

No external dependencies — uses stdlib urllib.
"""

import json
import random
import time
import urllib.request
import urllib.error
from pathlib import Path

COMFYUI_DEFAULT = 'http://localhost:8188'
NAS_LORA_DIR = Path('/mnt/orrigins/comfyui/models/loras')

# Output directory — symlinked to shared NAS folder so all machines
# share the same output browser. See infrastructure.md for details.
LOCAL_OUTPUT_DIR = Path('/home/corr/comfy/ComfyUI/output')
VAULT_OUTPUT_DIR = Path('/mnt/vault/watch/gen')

# ---------------------------------------------------------------------------
# ComfyUI API client
# ---------------------------------------------------------------------------

def _api_get(url: str, path: str, timeout: int = 10):
    """GET request to ComfyUI API."""
    try:
        req = urllib.request.Request(f'{url}{path}')
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def _api_post(url: str, path: str, data: dict, timeout: int = 30):
    """POST request to ComfyUI API."""
    try:
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            f'{url}{path}', data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def comfyui_available(url: str = COMFYUI_DEFAULT) -> bool:
    """Check if ComfyUI API is reachable."""
    result = _api_get(url, '/system_stats', timeout=3)
    return result is not None


def get_system_stats(url: str = COMFYUI_DEFAULT) -> dict | None:
    return _api_get(url, '/system_stats')


def list_checkpoints(url: str = COMFYUI_DEFAULT) -> list:
    """List available checkpoint models."""
    data = _api_get(url, '/object_info/CheckpointLoaderSimple')
    if not data:
        return []
    try:
        return data['CheckpointLoaderSimple']['input']['required']['ckpt_name'][0]
    except (KeyError, IndexError):
        return []


def list_loras(url: str = COMFYUI_DEFAULT) -> list:
    """List available LoRA models."""
    data = _api_get(url, '/object_info/LoraLoader')
    if not data:
        return []
    try:
        return data['LoraLoader']['input']['required']['lora_name'][0]
    except (KeyError, IndexError):
        return []


def list_samplers(url: str = COMFYUI_DEFAULT) -> list:
    """List available samplers."""
    data = _api_get(url, '/object_info/KSampler')
    if not data:
        return []
    try:
        return data['KSampler']['input']['required']['sampler_name'][0]
    except (KeyError, IndexError):
        return ['euler', 'euler_ancestral', 'dpmpp_2m', 'dpmpp_sde']


def list_schedulers(url: str = COMFYUI_DEFAULT) -> list:
    """List available schedulers."""
    data = _api_get(url, '/object_info/KSampler')
    if not data:
        return []
    try:
        return data['KSampler']['input']['required']['scheduler'][0]
    except (KeyError, IndexError):
        return ['normal', 'karras', 'sgm_uniform']


def queue_prompt(url: str, workflow: dict) -> str | None:
    """Submit a workflow to ComfyUI. Returns prompt_id."""
    result = _api_post(url, '/prompt', {'prompt': workflow})
    if result:
        return result.get('prompt_id')
    return None


def get_history(url: str, prompt_id: str) -> dict | None:
    return _api_get(url, f'/history/{prompt_id}')


def get_queue_status(url: str = COMFYUI_DEFAULT) -> dict:
    return _api_get(url, '/queue') or {}


def get_image_url(url: str, filename: str, subfolder: str = '',
                  img_type: str = 'output') -> str:
    """Build the URL to fetch an image from ComfyUI."""
    params = f'filename={filename}&type={img_type}'
    if subfolder:
        params += f'&subfolder={subfolder}'
    return f'{url}/view?{params}'


def fetch_image(url: str, filename: str, subfolder: str = '',
                img_type: str = 'output') -> bytes | None:
    """Fetch image bytes from ComfyUI."""
    try:
        img_url = get_image_url(url, filename, subfolder, img_type)
        req = urllib.request.Request(img_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError):
        return None


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_workflow(checkpoint: str, loras: list, prompt: str,
                   negative_prompt: str = '', seed: int = None,
                   steps: int = 20, cfg: float = 7.0,
                   width: int = 1024, height: int = 1024,
                   sampler: str = 'euler', scheduler: str = 'normal') -> dict:
    """Build a ComfyUI API workflow with checkpoint + LoRAs.

    Args:
        checkpoint: Checkpoint filename
        loras: List of dicts: [{name, model_strength, clip_strength}]
        prompt: Positive prompt text
        negative_prompt: Negative prompt text
        seed: Fixed seed (random if None)
        steps/cfg/width/height/sampler/scheduler: Generation params

    Returns:
        ComfyUI API format workflow dict (node_id -> node_spec)
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    workflow = {}
    node_id = 1

    # Node 1: CheckpointLoaderSimple
    ckpt_id = str(node_id)
    workflow[ckpt_id] = {
        'class_type': 'CheckpointLoaderSimple',
        'inputs': {'ckpt_name': checkpoint},
    }
    node_id += 1

    # Track current model/clip outputs (for chaining LoRAs)
    model_source = [ckpt_id, 0]  # MODEL output
    clip_source = [ckpt_id, 1]   # CLIP output
    vae_source = [ckpt_id, 2]    # VAE output

    # LoRA chain
    for lora in loras:
        lora_id = str(node_id)
        workflow[lora_id] = {
            'class_type': 'LoraLoader',
            'inputs': {
                'lora_name': lora['name'],
                'strength_model': lora.get('model_strength', 0.5),
                'strength_clip': lora.get('clip_strength', 0.5),
                'model': model_source,
                'clip': clip_source,
            },
        }
        model_source = [lora_id, 0]
        clip_source = [lora_id, 1]
        node_id += 1

    # Positive prompt
    pos_id = str(node_id)
    workflow[pos_id] = {
        'class_type': 'CLIPTextEncode',
        'inputs': {
            'text': prompt,
            'clip': clip_source,
        },
    }
    node_id += 1

    # Negative prompt
    neg_id = str(node_id)
    workflow[neg_id] = {
        'class_type': 'CLIPTextEncode',
        'inputs': {
            'text': negative_prompt or '',
            'clip': clip_source,
        },
    }
    node_id += 1

    # Empty latent image
    latent_id = str(node_id)
    workflow[latent_id] = {
        'class_type': 'EmptyLatentImage',
        'inputs': {
            'width': width,
            'height': height,
            'batch_size': 1,
        },
    }
    node_id += 1

    # KSampler
    sampler_id = str(node_id)
    workflow[sampler_id] = {
        'class_type': 'KSampler',
        'inputs': {
            'model': model_source,
            'positive': [pos_id, 0],
            'negative': [neg_id, 0],
            'latent_image': [latent_id, 0],
            'seed': seed,
            'steps': steps,
            'cfg': cfg,
            'sampler_name': sampler,
            'scheduler': scheduler,
            'denoise': 1.0,
        },
    }
    node_id += 1

    # VAE Decode
    decode_id = str(node_id)
    workflow[decode_id] = {
        'class_type': 'VAEDecode',
        'inputs': {
            'samples': [sampler_id, 0],
            'vae': vae_source,
        },
    }
    node_id += 1

    # Save Image
    save_id = str(node_id)
    workflow[save_id] = {
        'class_type': 'SaveImage',
        'inputs': {
            'images': [decode_id, 0],
            'filename_prefix': 'orracle_compare',
        },
    }

    return workflow


# ---------------------------------------------------------------------------
# Bake workflow — merge LoRA weights into a checkpoint permanently
# ---------------------------------------------------------------------------

def build_bake_workflow(checkpoint: str, lora: str,
                        model_weight: float, clip_weight: float,
                        output_prefix: str = 'forge_baked') -> dict:
    """Build a minimal workflow to bake a LoRA into a checkpoint.

    Pipeline: CheckpointLoaderSimple → LoraLoader → CheckpointSave
    The saved checkpoint appears in ComfyUI's output/checkpoints/ dir.
    """
    return {
        '1': {
            'class_type': 'CheckpointLoaderSimple',
            'inputs': {'ckpt_name': checkpoint},
        },
        '2': {
            'class_type': 'LoraLoader',
            'inputs': {
                'lora_name': lora,
                'strength_model': model_weight,
                'strength_clip': clip_weight,
                'model': ['1', 0],
                'clip': ['1', 1],
            },
        },
        '3': {
            'class_type': 'CheckpointSave',
            'inputs': {
                'model': ['2', 0],
                'clip': ['2', 1],
                'vae': ['1', 2],
                'filename_prefix': output_prefix,
            },
        },
    }


def poll_prompt_completion(url: str, prompt_id: str,
                           timeout: int = 600, interval: float = 2.0) -> dict | None:
    """Poll ComfyUI history until a prompt completes. Returns history entry or None."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = get_history(url, prompt_id)
        if history and prompt_id in history:
            return history[prompt_id]
        time.sleep(interval)
    return None


def extract_images(history_entry: dict) -> list:
    """Extract image info dicts from a completed history entry."""
    images = []
    for node_out in history_entry.get('outputs', {}).values():
        for img in node_out.get('images', []):
            images.append({
                'filename': img['filename'],
                'subfolder': img.get('subfolder', ''),
                'type': img.get('type', 'output'),
            })
    return images


# ---------------------------------------------------------------------------
# Output directory awareness
# ---------------------------------------------------------------------------

def get_output_info(output_dir: Path = None) -> dict:
    """Get output directory status including symlink target and file count."""
    out = output_dir or LOCAL_OUTPUT_DIR
    info = {
        'path': str(out),
        'exists': out.exists(),
        'is_symlink': out.is_symlink(),
        'target': None,
        'file_count': 0,
        'vault_connected': False,
    }
    if out.is_symlink():
        info['target'] = str(out.resolve())
        info['vault_connected'] = VAULT_OUTPUT_DIR.exists()
    if out.exists():
        info['file_count'] = sum(1 for f in out.iterdir()
                                 if f.suffix.lower() in ('.png', '.jpg', '.webp'))
    return info
