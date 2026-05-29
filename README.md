# flux2-eval

Multi-GPU serving and benchmarking for [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) image editing.

## What it does

- Serves a `/edit` endpoint that accepts an image + text prompt and returns an edited image via the FLUX.2-dev pipeline.
- Data-parallel inference across all available CUDA GPUs, one request per GPU at a time.
- Per-GPU LRU image cache (keyed by SHA-256) with cache-affinity routing — repeated edits of the same source image hit the same GPU and skip re-decoding.
- Includes a concurrency-sweep load-test harness that outputs structured JSON metrics.

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI app: GPU detection, model loading, caching, routing, `/edit` endpoint |
| `server.py` | Uvicorn entrypoint (port 8000, single worker) |
| `benchmark.py` | Load tester with single-prompt and multi-prompt modes |
| `setup.sh` | VM provisioning (venv, PyTorch+CUDA 12.9, `pip install -e .`) |

## Quick start

```bash
# Provision (requires CUDA GPUs)
bash setup.sh

# Set HuggingFace token for model download (~74 GB)
export HF_TOKEN=<your-token>

# Run server
source .venv/bin/activate
python server.py

# Benchmark
python benchmark.py --url http://localhost:8000/edit \
  --image test_image.png --prompt "Make it snowy" \
  --concurrency 1,2,4 --requests-per-level 5 --output results.json
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `FLUX2_MODEL_PATH` | HF model ID or local path | `black-forest-labs/FLUX.2-dev` |
| `VAE_CACHE_SIZE` | Per-GPU LRU cache entries | `10` |
| `HF_TOKEN` | HuggingFace auth token | — |

## Requirements

- CUDA GPUs (no CPU fallback)
- Python 3.10+
- PyTorch with CUDA 12.9
