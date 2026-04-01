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
