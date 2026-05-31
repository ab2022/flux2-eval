# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**flux2-eval** is a multi-GPU serving and benchmarking platform for FLUX.2-dev image editing, **designed for deployment on remote NVIDIA GPU hosts**. It runs a FastAPI server that accepts image + text prompt pairs, routes requests to available GPUs with cache-affinity optimization, and includes a load-testing harness for performance evaluation.

Workflow: Develop locally, rsync to remote GPU host, run there. This repo is **not meant to run locally** (no CPU fallback, requires VRAM >40 GB per GPU).

### Key Architecture

- **GPU Load Balancing**: Multi-GPU data parallelism with per-GPU LRU image cache (12KB lookups by SHA-256). Requests route to: (1) preferred GPU if pinned, (2) GPU with cached image, (3) first available, (4) wait for available.
- **Memory Optimization**: Automatic quantization (4-bit) for GPUs <70GB VRAM, CPU offload for full precision. Torch compile in `mode="default"` for CUDA compatibility with thread-safe executor.
- **Per-GPU Cache**: ImageCache holds PIL Images in CPU RAM keyed by SHA-256. Cache hits skip JPEG/PNG decode (vae_encode_time delta drives benchmark value).
- **Request Routing**: `route_request()` implements cache-affinity with semaphore-based backpressure across GPU slots.

## Remote Deployment & Commands

Typical workflow: develop locally → rsync to remote GPU host → execute on remote with localhost requests.

### Initial Setup on Remote GPU Host

```bash
# Clone repo and run provisioning (one-time)
git clone <repo> && cd flux2-eval
bash setup.sh                           # Provision venv, install PyTorch+CUDA 12.9, pip install -e .

# Set HuggingFace token
export HF_TOKEN=<your-hf-token>
```

### On Remote GPU Host (After Each Rsync)

```bash
# Start server (listens on localhost:8000)
source .venv/bin/activate
python server.py

# In another shell on the same remote host, run benchmarks
source .venv/bin/activate
python benchmark.py \
  --url http://localhost:8000/edit \
  --image path/to/test.png \
  --prompt "Your edit prompt" \
  --concurrency 1,2,4,8 \
  --requests-per-level 20 \
  --output results.json

# Multi-prompt benchmark (same image, different edits)
python benchmark.py \
  --url http://localhost:8000/edit \
  --image path/to/test.png \
  --prompts-file edits.txt \
  --concurrency 1,2,4 \
  --output results.json
```

## File Structure

| File | Purpose |
|---|---|
| `app.py` | Core FastAPI app: GPU detection, model loading, ImageCache, cache-affinity routing, `/edit` POST endpoint |
| `server.py` | Uvicorn entrypoint; runs single-worker on 0.0.0.0:8000 |
| `benchmarks/benchmark.py` | Async load tester; concurrency sweep, per-request metrics (queue latency, service time, cache hits), structured JSON output |
| `benchmarks/run_bench.py`, `run_serial.sh`, `decode_images.sh` | Utility scripts for benchmark orchestration |
| `setup.sh` | VM provisioning: venv setup, PyTorch+CUDA 12.9 install, GPU verification |

## Environment

- **Requires NVIDIA CUDA GPUs** (torch.cuda must report >0 devices; no CPU fallback; NVIDIA drivers required)
- **Python 3.10+**
- **CUDA 12.9** (PyTorch installed from `download.pytorch.org/whl/cu129`)
- **Not meant to run locally** — deploy to remote GPU host via rsync

### Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace auth token (required for model download, ~74 GB) |
| `FLUX2_MODEL_PATH` | `black-forest-labs/FLUX.2-dev` | Model ID or local path |
| `VAE_CACHE_SIZE` | `10` | Per-GPU LRU cache entries |
| `FLUX2_FORCE_QUANTIZE` | — | Set to `1`, `true`, or `yes` to force 4-bit quantization on all GPUs |

## Key Implementation Details

### GPU Memory Management

- **Quantized Pipeline** (~18–20 GB): Full pipeline on-device (GPU), compiled transformer, no CPU offload. Used when `vram_gb < 70` or forced.
- **Full Precision** (~74 GB): CPU offload strategy; only active models on-device. Used on high-VRAM GPUs.
- **Thread-Safe Inference**: Transformer compiled `mode="default"` (not `reduce-overhead`) to avoid CUDA graph crashes in multi-threaded executor.

### Benchmark Metrics

- `queue_latency_s`: Time from request submission to GPU acquire
- `service_time_s`: GPU inference time (includes VAE encode on cache miss)
- `vae_encode_time_s`: VAE encode time on cache miss; 0 on hit
- `cache_hit`: Boolean; true if image found in GPU's ImageCache, false on decode
- `gpu_id`: GPU index that handled the request

### Cache Behavior

- `compute_image_hash()` computes SHA-256 of raw byte stream (JPEG/PNG agnostic)
- Cache hit avoids decoding; input image pulled from CPU RAM, VAE encode still happens during pipeline inference
- On cache miss, decode happens inline; decoded PIL Image cached for subsequent requests
- LRU eviction when cache reaches `VAE_CACHE_SIZE` entries

## Testing & Debugging

- Start server with `python server.py` and watch logs for GPU load, model compile times, cache stats
- Single-request test: `curl -F "prompt=..." -F "image=@test.png" http://localhost:8000/edit`
- Benchmark output is structured JSON; pipe to `jq` for analysis (e.g., `jq '.level_summaries | map({concurrency, throughput_rps, latency_p99_s})'`)
- Common issues: No GPUs detected (check CUDA); model download fails (check `HF_TOKEN`); out of memory (enable quantization or reduce batch size)
