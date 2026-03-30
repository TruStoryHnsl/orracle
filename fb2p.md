# Orracle — Feedback to Prompt Queue

---

## Entry: 2026-03-27 19:30 — routed from orradash feedback pipeline

### Raw Input
> "Orracle doesn't expose APIs yet. Create APIs for image gen that let the endpoint control batch size, batch number, and profile to use for the prompt. Make sure these APIs are locked down to only be used by orradash. Another program may connect another time, but I want to make sure this API can't be used by any outside programs."

### Optimized Prompt

## Objective
Add a headless image generation API to orracle that can be called by orradash for dashboard-integrated image gen. The API must be auth-locked so only authorized clients (orradash initially) can use it.

## Requirements
1. New API endpoint: `POST /api/image/generate`
   - Request body:
     ```json
     {
       "prompt": "string",
       "batch_size": 1,       // images per batch
       "batch_count": 1,      // number of batches to run
       "profile": "default",  // named generation profile (model, sampler, steps, etc.)
       "negative_prompt": ""  // optional
     }
     ```
   - Response: job ID for async tracking, or direct result for small batches
2. New API endpoint: `GET /api/image/status/<job_id>` — poll job progress
3. New API endpoint: `GET /api/image/result/<job_id>` — retrieve generated images (base64 or URL)
4. New API endpoint: `GET /api/image/profiles` — list available generation profiles
5. Auth: API key header `X-Orracle-Key` checked on all `/api/image/*` endpoints
   - Key stored in orracle's `.env` as `IMAGE_API_KEY`
   - Same key configured in orradash's `.env` as `ORRACLE_IMAGE_KEY`
   - Reject all requests without valid key with 403

## Constraints
- Must work with the existing ComfyUI integration (`training/comfyui.py`, `blueprints/image_gen_bp.py`)
- Profiles are server-side configs — the client only sends the profile name, not raw ComfyUI parameters
- The existing web UI (`/studio/image`) should NOT require the API key (it uses Flask session auth)
- API key auth is separate from Flask-Login — headless clients don't have sessions

## Technical Decisions
- Add a new blueprint: `api_image_bp` at `/api/image/`
- Use the existing `JobQueue` for async job dispatch
- Profiles stored in `config/image_profiles.yaml` — each profile specifies: checkpoint, sampler, scheduler, steps, CFG, resolution
- Default profile ships with sensible defaults for the available ComfyUI setup
- The `@before_request` pattern checks `X-Orracle-Key` header on all `/api/image/*` routes

## Acceptance Criteria
- `POST /api/image/generate` with valid key queues a ComfyUI job and returns a job ID
- `GET /api/image/status/<id>` returns progress (queued/running/complete/failed)
- `GET /api/image/result/<id>` returns generated image data
- Requests without `X-Orracle-Key` header return 403
- Orradash can call these endpoints to generate and display images in a tile

### Status
Generated: 2026-03-27 19:30
Executed: 2026-03-27 22:30
Queued: 0 of 1
Tag: new_feature, api, security

### Implementation Notes
- Blueprint: `blueprints/api_image_bp.py` at `/api/image/`
- Profiles: `config/image_profiles.yaml` (default, quality, fast)
- Auth: `X-Orracle-Key` header, key from `IMAGE_API_KEY` env var
- Jobs route through existing `JobQueue` (SSE dashboard visibility)
- Also fixed: `comfyui.build_workflow()` now accepts batch_size/filename_prefix
- Also fixed: `job_queue._dispatch_image_gen()` was calling nonexistent `poll_prompt`

---

## Entry: 2026-03-27 22:45 — user feedback (image gen feature parity)

### Raw Input
> "The generation pages aren't feature complete. They both only have the shallowest of controls so they aren't all that useful. The image gen page needs to have all the options. At the very least it needs to have feature parity with orragen."

### Optimized Prompt
Bring orracle's `/studio/image` page to feature parity with orragen's image generation UI.

### Status
Generated: 2026-03-27 22:45
Executed: 2026-03-27 22:55
Tag: feedback, ui, feature_parity

### Implementation Notes
- Server-side workflow building (client sends params, server calls `build_workflow()`)
- Multi-LoRA management with model_strength + clip_strength sliders (add/remove)
- CLIP skip support (1-12, inserts CLIPSetLastLayer node)
- Batch size (1-8) + batch count (1-10)
- Auto Pony Tags toggle (quality tags + negative preset)
- Proper seed behavior (blank = random, not -1 hack)
- Image proxy endpoint (`/api/proxy`) — no CORS issues
- Working gallery with actual image display
- Lightbox viewer with arrow navigation + keyboard (Esc/Left/Right)
- Content blur toggle persisted in localStorage

### Still TODO vs orragen
- Profile save/load
- Exploration/grid mode (parameter sweeps)
- Session history panel
- Gallery tab with infinite scroll + filter + batch delete
- Feedback/rating system
- Upscale pass (1.5x, 2x)
- Auto-start ComfyUI on 503

---

## Entry: 2026-04-12 05:20 — routed from workspace feedback (2026-04-12 05:20.md)

### Raw Input
Source: `.feedback/2026-04-12 05:20.md` (42 lines)
> Architecture pattern: treat ComfyUI like a stateless worker process. A Job Queue Manager holds pending jobs, a Worker spawns ComfyUI per job on a random free port, streams progress via WebSocket/SSE, then kills the process group. Sequential queue, no parallelism. UI connects to the manager, not ComfyUI directly.

### Optimized Prompt

## Objective
Add a **stateless ComfyUI worker mode** to orracle's `job_queue.py` dispatch system. Currently, orracle connects to persistent ComfyUI instances on networked machines via `ServiceManager`. The new mode spawns a fresh ComfyUI process per image generation job and kills it after completion — guaranteeing clean GPU state, no VRAM leaks, and predictable resource usage.

## Requirements
1. **Ephemeral ComfyUI launcher** — new module or class (e.g., `training/comfyui_launcher.py` or extend existing `training/comfyui.py`) that:
   - Finds a free port via `socket.bind(('', 0))`
   - Spawns ComfyUI with `--port <free_port>` in a new process group (`preexec_fn=os.setpgrp`)
   - Polls `/system_stats` until ready (configurable timeout, default 60s)
   - Submits the workflow via existing ComfyUI client functions
   - Bridges ComfyUI's WebSocket progress → SSE events through orracle's existing SSE infrastructure
   - On completion/error, `os.killpg(pgid, signal.SIGTERM)` the entire process group
2. **JobQueue integration** — `_dispatch_image_gen()` in `job_queue.py` supports a per-machine `comfyui_mode` config:
   - `stateless`: uses ephemeral launcher (spawn → use → kill per job)
   - `persistent` (current default): connects to existing running ComfyUI instance via `ServiceManager`
3. **SSE lifecycle events** — image gen jobs emit: `spawning`, `ready`, `progress` (step/total), `node_executing`, `complete` (with output files), `error` — visible on the dashboard
4. **Machine-level config** — stateless vs persistent is set per machine in `config/machines.yaml`, since some nodes may run persistent ComfyUI while others use ephemeral mode
5. **Multi-machine port isolation** — random free port means multiple orracle workers can target the same machine without port conflicts

## Constraints
- Must integrate with orracle's existing `ServiceManager` and `JobQueue` architecture
- Must work with the existing SSE streaming infrastructure (dashboard real-time updates)
- `persistent` mode remains the default — no breaking changes
- ComfyUI runs on GPU machines (orrion etc.) — local-spawn mode first, then SSH-spawn for remote nodes
- The Studio image gen page (`/studio/image/`) and headless API (`/api/image/`) work identically regardless of mode

## Technical Decisions
- `subprocess.Popen` with `preexec_fn=os.setpgrp` for process group isolation
- Random port via `socket(AF_INET, SOCK_STREAM).bind(('', 0))` then close
- Config via `comfyui_mode: stateless|persistent` in `config/machines.yaml` per machine entry
- `COMFYUI_DIR` config per machine (path to ComfyUI installation on that node)
- Reuse existing `training/comfyui.py` functions for workflow submission — launcher wraps the lifecycle

## Open Questions
1. **Remote spawn**: orracle's compute nodes are networked machines (orrion). Does stateless mode initially only work when orracle and ComfyUI are on the same machine, or should SSH-based remote spawn be implemented from the start?
2. **GPU VRAM cleanup**: After `SIGTERM`, does ComfyUI reliably release VRAM? May need `SIGKILL` fallback after timeout.
3. **ServiceManager interaction**: Should stateless-mode machines still appear in ServiceManager health checks, or only when a job is running?

## Acceptance Criteria
- Setting `comfyui_mode: stateless` on a machine causes `job_queue.py` to spawn ComfyUI per image gen job and kill it after
- No orphan ComfyUI processes after job completion
- SSE stream shows full lifecycle events on the orracle dashboard
- Persistent mode (default) works unchanged
- Failed spawn cleans up and reports error in job status
- Documentation: docstrings on public functions, config format documented in `config/machines.yaml`

### Status
Generated: 2026-04-12 05:20
Executed: pending
Queued: 1 of 1
Tag: architecture, comfyui, worker

