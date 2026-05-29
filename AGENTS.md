# AGENTS.md

## Project

flux2-eval — FLUX.2-dev image editing server with data-parallel multi-GPU serving, per-GPU LRU image caching, and cache-affinity routing. Includes a load-test benchmark harness.

## Layout

- `app.py` — FastAPI app: GPU detection, model loading, cache-affinity routing, `/edit` endpoint
- `server.py` — uvicorn entrypoint (port 8000, single worker)
- `benchmark.py` — concurrency-sweep load tester (single-prompt and multi-prompt modes)
- `setup.sh` — VM provisioning script (hardcodes `/home/azureuser/flux2-eval`)

All application code lives in three top-level modules (`app`, `server`, `benchmark`). No packages or subdirectories.

## Commands

```bash
# Setup (on target VM with CUDA GPUs)
bash setup.sh            # creates .venv, installs PyTorch+CUDA 12.9, pip install -e .

# Run server
source .venv/bin/activate && python server.py

# Benchmark
python benchmark.py --url http://localhost:8000/edit \
  --image test_image.png --prompt "..." \
  --concurrency 1,2,4 --requests-per-level 3 --output results.json

# Benchmark with prompt file (related-request / cache-affinity mode)
python benchmark.py --url http://localhost:8000/edit \
  --image test_image.png --prompts-file sample_prompts.txt \
  --concurrency 1,2,4 --output results.json
```

## Prerequisites

- **CUDA GPUs required** — no CPU fallback; server crashes at startup without GPUs
- **HF_TOKEN** env var needed for HuggingFace model download (`black-forest-labs/FLUX.2-dev`)
- **~74 GB model weights** — uses `enable_model_cpu_offload` to fit in 80 GB VRAM per GPU
- Python 3.10+, PyTorch with CUDA 12.9

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `FLUX2_MODEL_PATH` | HF model ID or local path | `black-forest-labs/FLUX.2-dev` |
| `VAE_CACHE_SIZE` | Per-GPU LRU cache entries | `10` |
| `HF_TOKEN` | HuggingFace auth token | (none) |

## Architecture notes

- Single uvicorn worker (`workers=1`); concurrency via async + `ThreadPoolExecutor` (one thread per GPU)
- Per-GPU `asyncio.Semaphore(1)` — only one inference per GPU at a time
- Routing: prefer idle GPU with cached source image, then first-available, then backpressure wait
- Image cache keyed by SHA-256 of raw upload bytes; cache lives in CPU memory

## No tests, CI, or linting

There are no test files, CI workflows, or linter/formatter configs. Changes should be verified by reading the code and, when possible, running the server manually.
