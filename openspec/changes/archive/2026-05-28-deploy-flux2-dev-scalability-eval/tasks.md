## 1. Project Setup

- [x] 1.1 Create Python project structure with `pyproject.toml` declaring dependencies: `fastapi`, `uvicorn`, `diffusers`, `transformers`, `accelerate`, `torch`, `bitsandbytes`, `aiohttp`
- [x] 1.2 Create a virtual environment and install dependencies with CUDA 12.9 support (`--extra-index-url https://download.pytorch.org/whl/cu129`)
- [x] 1.3 Accept the FLUX.2-dev Non-Commercial License on HuggingFace and verify model download access (or set `FLUX2_MODEL_PATH` to pre-downloaded weights)

## 2. Model Loading and GPU Detection

- [x] 2.1 Implement GPU detection: enumerate available CUDA GPUs and query VRAM per device using `torch.cuda.get_device_properties()`
- [x] 2.2 Implement model loading function that loads one FLUX.2-dev diffusers pipeline per GPU — bf16 if VRAM >= 70GB, bitsandbytes 4-bit quantization otherwise
- [x] 2.3 Support `FLUX2_MODEL_PATH` environment variable for loading from a local path instead of HuggingFace Hub
- [x] 2.4 Configure a `ThreadPoolExecutor(max_workers=NUM_GPUS)` and set it as the default executor on the event loop

## 3. VAE Latent Cache

- [x] 3.1 Implement a per-GPU LRU cache class for VAE-encoded latent tensors, keyed by SHA-256 hex string, with configurable max size (default 10)
- [x] 3.2 Implement SHA-256 hashing of raw image bytes for cache key generation
- [x] 3.3 Integrate cache into the inference path: check cache before VAE encoding, store result after encoding, track `vae_encode_time_s` and `cache_hit` per request

## 4. Request Routing

- [x] 4.1 Implement per-GPU `asyncio.Semaphore(1)` for concurrency control
- [x] 4.2 Implement cache-affinity routing: given an `image_hash`, check which idle GPU (if any) holds that hash in its latent cache; prefer that GPU, fall back to first-available scan by index
- [x] 4.3 Implement backpressure: if all GPUs are busy, await until a semaphore is released

## 5. HTTP Endpoint

- [x] 5.1 Create FastAPI app with `POST /edit` endpoint accepting multipart form: `image` (file, optional if `image_hash` provided), `prompt` (string, required), `image_hash` (string, optional), `num_steps` (int, default 28), `guidance_scale` (float, default 4.0)
- [x] 5.2 Implement request validation: return 422 if prompt is missing; return 404 if `image_hash` provided without image file and hash not found in assigned GPU's cache
- [x] 5.3 Implement the inference pipeline: route to GPU → check VAE cache → encode if miss → run denoising → decode output → return base64 PNG + metrics JSON
- [x] 5.4 Implement per-request metrics collection using `time.monotonic()`: `request_id` (UUID), `gpu_id`, `queue_latency_s`, `service_time_s`, `vae_encode_time_s`, `total_time_s`, `num_steps`, `cache_hit`, `image_hash` — all rounded to 3 decimal places
- [x] 5.5 Add startup event to load all models before accepting requests; log GPU count, VRAM, and quantization mode per GPU

## 6. Server Launch

- [x] 6.1 Create `server.py` entry point that starts uvicorn with `--host 0.0.0.0 --port 8000 --workers 1`
- [x] 6.2 Verify server starts, loads 4 model replicas, and responds to a single `curl` request with correct JSON structure

## 7. Benchmark Harness — Core

- [x] 7.1 Create `benchmark.py` with CLI argument parsing: `--url`, `--image`, `--prompt`, `--concurrency` (comma-separated), `--requests-per-level`, `--output`
- [x] 7.2 Implement async request sender using `aiohttp`: multipart form POST with image file, prompt, and num_steps; parse response JSON for server-side metrics
- [x] 7.3 Implement concurrency-level runner: use `asyncio.Semaphore` to cap concurrent requests; run all requests at one level before proceeding to the next
- [x] 7.4 Implement per-request metric recording: `request_id`, `concurrency_level`, `submit_time`, `response_time`, `total_latency_s`, `queue_latency_s`, `service_time_s`, `vae_encode_time_s`, `cache_hit`, `gpu_id`, `status_code`, `error`
- [x] 7.5 Implement 600-second per-request timeout with failure recording on timeout

## 8. Benchmark Harness — Related Requests and Cache Benchmarking

- [x] 8.1 Add `--prompts-file` CLI argument: read one prompt per line from a text file
- [x] 8.2 Implement related-request mode: send first request with image upload, capture `image_hash` from response, send subsequent requests with `image_hash` only (no image file)
- [x] 8.3 Create a sample prompts file with 10+ related editing prompts for testing (e.g., logo text variations on the same hat image)

## 9. Benchmark Harness — Reporting

- [x] 9.1 Implement per-level summary computation: total/successful/failed counts, throughput (req/s), latency p50/p95/p99, mean queue latency, mean service time, GPU utilization spread, cache hit/miss counts
- [x] 9.2 Implement cache-split reporting: separate mean service time for cache-hit vs cache-miss requests
- [x] 9.3 Implement stdout summary printer (formatted table per concurrency level)
- [x] 9.4 Implement JSON output writer: one entry per concurrency level with `summary` and `requests` array; write to path specified by `--output`

## 10. End-to-End Validation

- [x] 10.1 Run a single-request smoke test via `curl` and verify response contains all expected metrics fields including `cache_hit`, `vae_encode_time_s`, and `image_hash`
- [x] 10.2 Run two sequential `curl` requests with the same image (second using `image_hash` from first response) and verify `cache_hit: true` on the second
- [x] 10.3 Run `benchmark.py` with `--concurrency 1,2,4 --requests-per-level 5` and verify JSON output structure and summary output
- [x] 10.4 Run `benchmark.py` in related-request mode with `--prompts-file` and verify cache hit rate is reported correctly
