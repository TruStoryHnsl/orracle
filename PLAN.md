# Orracle Development Plan

## Vision
Orracle is a fully feature-complete AI toolkit with two halves:
- **Studio**: Versatile generation environment for text, image, and video AI
- **Workshop**: Data processing pipeline and model training for any AI model type

The application connects to compute nodes on networked devices, providing unified control of ComfyUI, Ollama, and SD Forge instances across the cluster.

## Architecture (Current)

### Backend
- `app.py` — Flask app factory, inits ServiceManager + JobQueue
- `shared.py` — Consolidated config loading, hardware cache
- `services.py` — Service lifecycle manager (health, start/stop via SSH)
- `job_queue.py` — Unified job queue with routing, failover, persistence
- `blueprints/` — 9 blueprints (dashboard, text_gen, image_gen, forge, pipeline, training, export, machines, compare[legacy])
- `training/` — jobs, hardware, remote, export_mgr, generate, forge, comfyui, log_parser
- `executor/` — DAG pipeline engine (runner, preview, remote dispatch)
- `nodes/` — 20+ processing nodes (text, video, encoding)

### Frontend
- Kinetic Forge design system (Space Grotesk + Inter + JetBrains Mono, cyan accent, 0px radius)
- Sidebar navigation (desktop) + bottom tabs (mobile)
- SSE streaming for real-time dashboard updates

### Route Map
```
/                        Dashboard — command center
/studio/text/            Text generation (Ollama chat)
/studio/image/           Image generation (ComfyUI)
/studio/forge/           LoRA Forge (evolutionary refinement)
/workshop/pipeline/      Data pipeline + node editor
/workshop/train/         Training job management (Commander)
/workshop/export/        Export + deploy (fuse → GGUF → Ollama)
/machines/               Machine + service management
```

---

## Queued Features

### Headless Image Generation API (for orradash integration)
**Source**: orradash feedback pipeline 2026-03-27 | **Status**: DONE | **fb2p.md**: entry #1

New authenticated API at `/api/image/` for headless image generation:
- `POST /api/image/generate` — batch_size, batch_count, profile, prompt
- `GET /api/image/status/<id>` — job progress
- `GET /api/image/result/<id>` — retrieve images
- `GET /api/image/profiles` — list profiles

Auth: `X-Orracle-Key` header (API key in `.env`). Initially only orradash is authorized.

## Development Phases

### Phase A: Job Scheduler + Compute Load Watcher [DONE]
**Source: feedback item 1.5**

1. **Pre-planned job configs** — `plan_job()` in job_queue.py, persists in queue.yaml, visible on dashboard as "waiting"
2. **Suspend/resume** — `suspend()`/`resume()` with SIGSTOP/SIGCONT via SSH
3. **Compute load watcher** — `ComputeWatcher` class in services.py, polls nvidia-smi + CPU load every 15s, auto-throttles jobs with `throttle=True`
4. **Dashboard quick-start** — Waiting jobs show with Start button, throttle toggle on all job cards
5. **Throttle toggle API** — `POST /api/queue/<id>/throttle`, SSE broadcasts load events

### Phase B: Content Safety Layer [DONE]
**Source: feedback items 3, 4, 6**

1. **Blurred output** — Blur toggle on both `/studio/image` and `/studio/forge`, shared localStorage key
2. **Guardrail training type** — "Safety Guardrail Fine-Tune" option in training type dropdown
3. **Model quarantine** — `visibility: private` on all model_registry.yaml profiles, export page warns before deploying private models
4. **ComfyUI output isolation** — See Phase E

### Phase C: Jailbreak Testing Suite [DONE]
**Source: feedback item 5**

1. **Test library** — `config/jailbreak_tests.yaml` with 17 prompts across 6 categories (DAN, ignore-previous, roleplay, encoding, persona, multi-turn)
2. **Automated runner** — `training/audit.py` sends prompts via Ollama, keyword-based pass/fail, saves results to `config/audit_results/`
3. **Report page** — `/workshop/audit` with model selector, category checkboxes, SSE-streamed results table, pass/fail summary cards
4. **Red team profiles** — 6 attack categories as checkbox filters

### Phase D: Alternative Training Frameworks [DONE]
**Source: feedback item 2**

1. **Training types expanded** — 6 types in dropdown: CPT, SFT, Guardrail (active) + Classification, Embedding, Reward Model (planned/disabled)
2. **Config presets** — `config/presets/` has placeholder YAMLs for classification, embedding, reward model with suggested params and data format docs
3. **UI distinction** — Disabled options styled italic/muted, toast notification if somehow selected

### Phase G: Mobile Polish [DONE]
**Source: design spec**

1. **375px testing** — All pages tested via Playwright at 375px (dashboard, image gen, text gen, audit)
2. **Bottom tabs** — Font size reduced on mobile for proper spacing, 44px min touch targets
3. **Overflow prevention** — `overflow-x: hidden` on mobile page-body
4. **Touch targets** — Buttons and interactive elements meet 44px minimum
5. **Checkbox alignment** — Audit page checkboxes fixed with `width: auto` override

### Phase E: ComfyUI Output Isolation [DONE]
**Source: feedback item 8**

1. `build_workflow()` defaults to `filename_prefix='orracle'` — all orracle images namespaced under `orracle/` in ComfyUI output
2. Headless API uses `'orracle_api'` prefix
3. Forge inherits default `'orracle'` prefix
4. Image proxy serves from ComfyUI history (knows exact filenames)
5. Only ComfyUI's own web UI writes to its default output namespace

### Phase F: Dashboard Telemetry [DONE]
**Source: design spec gaps**

1. **showToast()** — Added missing global toast function to orracle.js (prerequisite for all pages)
2. **Hardware telemetry in node cards** — GPU util, VRAM usage, CPU load bars in each machine card, live-updated via SSE compute_load events
3. **Progress bar gradient** — Already uses var(--kinetic-gradient)
4. **Tabular-lining numbers** — font-variant-numeric: tabular-nums on .stat-value, .metric-value, .telemetry-val

---

## Dataset Philosophy
**Source: feedback item 1**

The nifty archive dataset is treated as an anthropological corpus — a comprehensive, uncensored record of publicly available amateur queer erotica. The training approach is:
1. **Phase 1**: Train uncensored base model on full dataset (current orrvert training)
2. **Phase 2**: Apply post-training guardrails (safety SFT pass)
3. **Phase 3**: Red-team test the guardrailed model
4. **Phase 4**: Delete source dataset — the model IS the distillation

The raw data is temporary. The trained model is the artifact. Guardrails are applied after training, not by filtering the training data.

## Model Distribution Policy
**Source: feedback item 7**

- All models are **private by default** — never auto-publish to HuggingFace or any model hub
- Export page must confirm before any upload/share action
- `model_registry.yaml` should track visibility: `private`, `local-only`, `shared`
- Orracle should never push models to external services without explicit user confirmation

---

## Recent Changes
- **2026-04-12**: Added stateless ComfyUI worker mode to roadmap (from workspace feedback pipeline)

## Completed Work (This Session)
- [x] Phase 1-7: Backend foundations (shared.py, services.py, job_queue.py)
- [x] CSS restructure: Kinetic Forge design system (8 CSS files)
- [x] Navigation: Sidebar (desktop) + bottom tabs (mobile)
- [x] Dashboard: Command center with stat cards, node cards, services, SSE
- [x] Studio: Text gen (chat), Image gen (ComfyUI), Forge (relocated)
- [x] Workshop: Pipeline, Commander, Export (relocated with new URLs)
- [x] Redirects: All old URLs → 301 redirects
- [x] Template migration: base.html updated with sidebar nav

## Next Priority

### Stateless ComfyUI Worker Mode
**Source**: workspace feedback 2026-04-12 | **Status**: planned | **fb2p.md**: entry #3

Spawn a fresh ComfyUI process per image gen job and kill it after completion. Config per machine in `config/machines.yaml` (`comfyui_mode: stateless|persistent`). Random free port, process group isolation, SSE lifecycle events. Persistent mode remains default.

Key files: `job_queue.py` (dispatch), `training/comfyui.py` (client), new `training/comfyui_launcher.py` (lifecycle).

Open questions: remote SSH-spawn timing, VRAM cleanup on SIGTERM, ServiceManager interaction for stateless machines.

### Remaining Studio/Image Gen Work
- Exploration/grid mode (parameter sweeps)
- Session history panel
- Gallery with infinite scroll + filter + batch delete
- Profile save/load
- Upscale pass (1.5x, 2x)
- Feedback/rating system
- Auto-start ComfyUI on 503
