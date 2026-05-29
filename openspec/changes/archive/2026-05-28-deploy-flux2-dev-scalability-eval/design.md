## Context

FLUX.2-dev is a 32B parameter rectified flow transformer from Black Forest Labs that supports text-to-image generation and instruction-based image editing. We are building a minimal serving and benchmarking stack to evaluate its scalability on a 4-GPU server (either 4x A100 80GB or 4x L40s 48GB).

There is no existing serving infrastructure in this project. The official BFL inference repo provides a CLI script (`scripts/cli.py`) and diffusers integration, but no HTTP serving layer or performance tooling. We are building both from scratch using standard Python libraries.

The model accepts a source image plus a natural-language editing instruction (e.g., "Change the logo on the hat to say 'LA'") and produces an edited image. Inference involves ~28-50 denoising steps through the transformer, making each request GPU-bound for several seconds. In typical usage, multiple related requests will target the same source image with different editing prompts (e.g., changing a logo to 'LA', then 'Bills', then 'OR'), making VAE encoding of the source image a cacheable operation.

## Goals / Non-Goals

**Goals:**
- Stand up an HTTP service that can serve image-editing requests against FLUX.2-dev on 4 GPUs simultaneously
- Maximize throughput via data parallelism (one model replica per GPU)
- Reduce per-request latency for related-request sequences by caching VAE-encoded source image latents per GPU
- Collect per-request metrics (queue latency, service time, GPU ID) and aggregate benchmarks (throughput, latency percentiles, saturation point)
- Support both A100 (bf16) and L40s (4-bit quantized) deployments with minimal code divergence
- Produce a repeatable benchmark harness that sweeps concurrency levels and outputs structured results

**Non-Goals:**
- Production hardening (auth, TLS, rate limiting, health checks, graceful shutdown)
- Model fine-tuning or quality evaluation (we measure speed, not output fidelity)
- Multi-node distributed serving (single server with 4 GPUs only)
- Tensor parallelism or pipeline parallelism (data parallelism is sufficient and simpler for the initial evaluation; tensor parallelism within NVLink pairs is a future follow-up)
- Persistent request queuing or job scheduling (in-memory semaphores are sufficient for benchmarking)
- Prompt upsampling or any pre/post-processing beyond basic image encode/decode

## Decisions

### 1. Serving framework: FastAPI + uvicorn (single process)

**Choice**: Single-process FastAPI app with async request handling, offloading GPU inference to a thread pool.

**Alternatives considered**:
- **vLLM**: Designed for LLM token generation, not diffusion models. No native diffusers pipeline support.
- **Triton Inference Server**: More capable but far more complex to set up. Overkill for an evaluation.
- **Ray Serve**: Good scaling story but adds a distributed runtime dependency we don't need on a single node.
- **Multiple uvicorn workers**: Would require each worker to load its own models, wasting VRAM. A single process managing all 4 GPU pipelines is simpler and more memory-efficient.

**Rationale**: FastAPI is the simplest path to an HTTP endpoint. A single process avoids the complexity of inter-process GPU assignment. The async event loop handles concurrent requests while `run_in_executor` offloads blocking inference to threads.

### 2. Parallelism: Data parallelism with one replica per GPU

**Choice**: Load an independent model replica onto each of the 4 GPUs. Each request is routed to whichever GPU is free.

**Alternatives considered**:
- **Tensor parallelism** (split model layers across GPUs): Reduces per-request latency by using multiple GPUs for one request, but halves max concurrency from 4 to 2. Requires NCCL communication between GPUs on every layer — viable only over NVLink, not PCIe.
- **Pipeline parallelism** (split model stages across GPUs): Adds pipeline bubbles and inter-GPU synchronization overhead. Only useful when a single GPU can't hold the model.

**Rationale**: Image-editing requests are independent — there's no shared state between them. Data parallelism gives linear throughput scaling (4x) with zero inter-GPU communication. On A100, the model fits in bf16 on a single GPU. On L40s, 4-bit quantization makes it fit.

**Future work — tensor parallelism within NVLink pairs**: The Azure Standard_NC96ads_A100_v4 instance has 2 NVLink-connected GPU pairs (GPU 0+1, GPU 2+3 at 600 GB/s). A hybrid configuration — tensor parallelism within each NVLink pair, data parallelism between pairs — would yield 2 worker slots instead of 4, but with ~1.5-1.8x faster per-request latency. This trades throughput for latency and may be preferable for latency-sensitive workloads. The benchmark harness can evaluate both configurations with no changes; only the model loading and worker pool size differ. This will be tried after the initial data-parallel evaluation establishes baseline numbers.

### 3. Request routing: Semaphore-per-GPU with cache-affinity preference

**Choice**: Each GPU has an `asyncio.Semaphore(1)`. When a request arrives, the router first checks whether any idle GPU already holds the source image's VAE latents in its cache (matched by SHA-256 hash of the image bytes). If so, the request is routed to that GPU. Otherwise, GPUs are scanned in index order and the first available semaphore is acquired. If all GPUs are busy, the request awaits until one frees up.

**Alternatives considered**:
- **Strict round-robin**: Doesn't account for variable inference times or cache locality; would scatter related requests across GPUs, defeating VAE caching.
- **Sticky sessions (always route same image to same GPU)**: Maximizes cache hits but can cause load imbalance if one source image dominates traffic. The affinity-with-fallback approach is a better tradeoff.
- **Dedicated asyncio.Queue per GPU**: More complex, similar behavior to semaphores for this use case.
- **External load balancer (nginx, HAProxy)**: Unnecessary for a single-process benchmark setup; can't make cache-aware routing decisions.

**Rationale**: Cache-affinity routing maximizes VAE latent reuse for related-request sequences without sacrificing utilization. When the preferred GPU is busy, falling back to first-available ensures no GPU sits idle. For 4 GPUs this lookup is trivially fast.

### 4. GPU memory strategy: bf16 on A100, bitsandbytes 4-bit on L40s

**Choice**: Detect available VRAM at startup. If >= 70GB per GPU, load bf16. Otherwise, load with `BitsAndBytesConfig(load_in_4bit=True)`.

**Alternatives considered**:
- **GPTQ quantization**: Requires a pre-quantized checkpoint and calibration data. More effort than bnb.
- **AWQ quantization**: Similar story to GPTQ; better for LLMs than diffusion models currently.
- **FP8 (on H100/L40s)**: L40s lacks FP8 Tensor Cores; H100 supports it but isn't our target hardware.
- **CPU offloading**: Would make inference unacceptably slow for benchmarking.

**Rationale**: bitsandbytes 4-bit is the easiest quantization path — it requires no pre-quantized weights and works directly with `from_pretrained`. HuggingFace also provides `diffusers/FLUX.2-dev-bnb-4bit` as a pre-quantized checkpoint. The quality/speed tradeoff is acceptable for scalability evaluation.

### 5. Benchmark design: Concurrency sweep with structured output

**Choice**: The benchmark harness runs N requests at each concurrency level (e.g., 1, 2, 4, 8, 16), collects per-request metrics from both client and server, and outputs results as JSON.

**Alternatives considered**:
- **wrk / hey / k6**: Generic HTTP load testers. They can't parse per-request server-side metrics (queue latency, GPU ID) from response bodies. They also can't send multipart form data with images easily.
- **Locust**: Could work but adds a dependency and web UI we don't need. Custom async Python gives us full control over metric collection.

**Rationale**: A custom `asyncio + aiohttp` script is ~150 lines, gives full control over request construction (multipart image upload), and can parse both client-side and server-side metrics into a unified report. The concurrency sweep pattern directly answers the scalability question: "at what concurrency does throughput plateau and queue latency spike?"

### 6. Inference parameters: 28 steps, guidance_scale=4.0

**Choice**: Default to 28 denoising steps with guidance scale 4.0 for benchmarking.

**Rationale**: BFL documentation indicates 28 steps is a good quality/speed tradeoff (vs. 50 for maximum quality). Since we're measuring throughput, not output quality, faster inference per request lets us run more benchmark iterations. The guidance scale of 4.0 is BFL's recommended default. These are configurable per-request via the API.

### 7. VAE latent caching for related-request sequences

**Choice**: Each GPU maintains an LRU cache (default 10 entries) of VAE-encoded latent tensors, keyed by SHA-256 hash of the source image bytes. When a request targets an already-cached source image, VAE encoding is skipped entirely. The server returns the `image_hash` in every response so clients can send it on subsequent requests (optionally omitting the image upload).

**Alternatives considered**:
- **No caching (recompute VAE on every request)**: Simpler, but wasteful for the expected workload where many edits target the same source image. VAE encoding is not the dominant cost (denoising is), but it's non-trivial and easily avoided.
- **Shared cache across GPUs (CPU-side)**: Would require transferring latent tensors between CPU and GPU memory on every request. The per-GPU approach keeps tensors on-device with zero transfer cost.
- **Disk-based cache**: Adds I/O latency and complexity. The latent tensors are small (~tens of MB) and fit comfortably in GPU VRAM alongside the model.

**Rationale**: The expected workload is sequences of related edits to the same source image (e.g., trying different logo texts on the same hat photo). Caching the VAE latents eliminates redundant encoding after the first request. Per-GPU caching is the simplest approach — no cross-GPU coordination, no CPU↔GPU transfers, and the LRU policy bounds VRAM consumption. Combined with cache-affinity routing (Decision 3), related requests naturally cluster on the same GPU to maximize hit rates.

## Risks / Trade-offs

- **[VRAM OOM on L40s]** → 4-bit quantization reduces weights to ~16GB but activations during inference could still spike. Mitigation: monitor VRAM usage during initial test runs; reduce batch size or image resolution if OOM occurs.

- **[Single-process GIL contention]** → Python's GIL could bottleneck when 4 threads run inference simultaneously. Mitigation: PyTorch releases the GIL during CUDA kernel execution, so actual GPU compute is not affected. Only Python-level pre/post-processing (image encode/decode) holds the GIL, and this is negligible compared to inference time.

- **[Quantization quality degradation]** → 4-bit models produce slightly different outputs than bf16. Mitigation: acceptable for benchmarking. If quality comparison is needed later, run a small set of identical prompts on both configurations and compare outputs visually.

- **[Model download time and license]** → FLUX.2-dev requires accepting the Non-Commercial License on HuggingFace before download. The 32B model is ~64GB. Mitigation: pre-download weights and set `FLUX2_MODEL_PATH` environment variable. Cache on local SSD.

- **[Benchmark variance]** → GPU thermal throttling and OS scheduling can cause run-to-run variance. Mitigation: run each concurrency level with >= 20 requests; report percentiles (p50/p95/p99) not just means; allow warm-up requests before measurement.

- **[Thread pool sizing]** → Default `run_in_executor(None, ...)` uses Python's default thread pool. With 4 GPUs we need at least 4 threads. Mitigation: explicitly set executor to `ThreadPoolExecutor(max_workers=NUM_GPUS)` at startup.

- **[VAE cache VRAM pressure]** → Cached latent tensors consume GPU VRAM alongside model weights and activations. Mitigation: LRU cache with a configurable max size (default 10 entries). Each cached latent is ~tens of MB, so 10 entries add <1GB VRAM — negligible compared to the ~64GB model. On L40s with tighter VRAM, reduce the cache size if OOM occurs.

- **[Cache-affinity routing imbalance]** → If most requests target the same source image, cache affinity could funnel traffic to one GPU while others idle. Mitigation: affinity is a preference, not a requirement — if the preferred GPU is busy, the request goes to the first available GPU. Under high concurrency, load naturally spreads across all GPUs.
