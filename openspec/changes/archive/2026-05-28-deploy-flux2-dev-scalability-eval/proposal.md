## Why

We need to evaluate FLUX.2-dev (32B parameter image editing model) for scalability before committing to it as a production image-editing backend. The model has no published serving benchmarks, and we need concrete numbers — throughput, latency distribution, saturation point — on our target hardware (4x L40s or 4x A100 GPUs) to make informed capacity-planning decisions.

## What Changes

- Add a FastAPI-based HTTP serving layer that wraps the FLUX.2-dev diffusers pipeline, with one model replica per GPU (data parallelism)
- Add a GPU-aware request router that distributes incoming image-edit requests across 4 GPU workers with backpressure via semaphores and cache-affinity routing
- Add an `/edit` HTTP endpoint that accepts a source image (PNG/JPG) and English editing instructions, returning the edited image plus per-request metrics
- Add per-GPU VAE latent caching so that related requests editing the same source image skip redundant VAE encoding after the first (cold-start) request
- Add server-side instrumentation that reports queue latency, service time, VAE encode time, cache hit/miss, GPU assignment, and request ID on every response
- Add an async load-testing harness that sweeps across concurrency levels, collects per-request metrics, and produces aggregate summaries (throughput, p50/p95/p99 latency, GPU spread)
- Add quantization support (bitsandbytes 4-bit) for L40s deployment where 48GB VRAM cannot fit the full bf16 model

## Capabilities

### New Capabilities

- `model-serving`: HTTP service that loads FLUX.2-dev onto available GPUs and serves image-editing requests via a REST API, with per-GPU VAE latent caching for related-request sequences
- `gpu-parallelism`: Data-parallel GPU worker pool with per-GPU semaphores, cache-affinity routing (prefer GPUs holding cached latents), and quantization fallback for memory-constrained GPUs. (Future: evaluate tensor parallelism within NVLink pairs on Azure NC96ads_A100_v4 to trade concurrency for lower per-request latency.)
- `metrics-collection`: Server-side per-request instrumentation (queue latency, service time, VAE encode time, cache hit/miss, GPU ID) and client-side end-to-end latency tracking
- `load-testing`: Async benchmark harness that drives concurrent requests at configurable levels, supports related-request sequences (same image, multiple prompts) to measure VAE cache effectiveness, and produces throughput/latency/utilization reports

### Modified Capabilities

_(none — this is a greenfield evaluation project)_

## Impact

- **New dependencies**: `fastapi`, `uvicorn`, `diffusers`, `transformers`, `accelerate`, `torch`, `bitsandbytes`, `aiohttp`
- **Hardware**: Requires a server with 4x NVIDIA GPUs (A100 80GB for bf16, or L40s 48GB with 4-bit quantization)
- **HuggingFace access**: Must accept FLUX.2-dev Non-Commercial License to download model weights
- **Network**: Service exposes port 8000; benchmark harness connects over HTTP
- **Artifacts produced**: `results.json` with per-request metrics and per-level summaries; stdout reports during benchmark runs
- **Future consideration**: The Azure Standard_NC96ads_A100_v4 instance has 2 NVLink-connected GPU pairs (GPU 0+1, GPU 2+3). A future iteration could benchmark tensor parallelism within each NVLink pair (2 replicas of 2 GPUs each) to reduce per-request latency at the cost of max concurrency (2 vs 4). This requires no API changes — only the model loading and worker pool configuration would change.
