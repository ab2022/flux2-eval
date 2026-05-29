## ADDED Requirements

### Requirement: Data-parallel GPU worker pool

The server SHALL maintain one model replica per available CUDA GPU. Each GPU SHALL process at most one inference request at a time, enforced by a per-GPU semaphore.

#### Scenario: 4 concurrent requests on 4 GPUs

- **WHEN** 4 requests arrive simultaneously on a server with 4 GPUs
- **THEN** each request is assigned to a different GPU and all 4 execute concurrently with no queuing

#### Scenario: 5th request queues

- **WHEN** a 5th request arrives while all 4 GPUs are busy
- **THEN** the request waits until a GPU becomes free, then is dispatched to that GPU

### Requirement: First-available GPU routing with cache affinity

The server SHALL prefer routing a request to a GPU that already has the request's source image latents cached. If an `image_hash` is provided (or computed from the uploaded image), the server SHALL first check whether any idle GPU holds that hash in its latent cache. If a cache-holding GPU is idle, the request SHALL be routed there. If no cache-holding GPU is idle, the server SHALL fall back to first-available routing (scanning GPUs in index order).

#### Scenario: Cache-affinity hit — preferred GPU is idle

- **WHEN** a request arrives with `image_hash` "abc123" and GPU 2 holds "abc123" in its cache and is idle
- **THEN** the request is routed to GPU 2, regardless of whether lower-indexed GPUs are also idle

#### Scenario: Cache-affinity miss — preferred GPU is busy

- **WHEN** a request arrives with `image_hash` "abc123" and GPU 2 holds "abc123" but is busy, while GPU 0 is idle
- **THEN** the request is routed to GPU 0 (first available), resulting in a cache miss on GPU 0

#### Scenario: No cache affinity — new image

- **WHEN** a request arrives with an image not cached on any GPU and GPU 0 and GPU 2 are both idle
- **THEN** the request is routed to GPU 0 (lowest index, first-available fallback)

### Requirement: Automatic quantization for memory-constrained GPUs

The server SHALL detect available VRAM per GPU at startup. If a GPU has >= 70GB VRAM, the server SHALL load the model in bf16. If a GPU has < 70GB VRAM, the server SHALL load the model with bitsandbytes 4-bit quantization.

#### Scenario: A100 80GB GPU

- **WHEN** the server starts and detects a GPU with 80GB VRAM
- **THEN** the model is loaded in bf16 (torch.bfloat16) on that GPU

#### Scenario: L40s 48GB GPU

- **WHEN** the server starts and detects a GPU with 48GB VRAM
- **THEN** the model is loaded with BitsAndBytesConfig(load_in_4bit=True) on that GPU

### Requirement: Thread pool executor for GPU inference

The server SHALL offload blocking GPU inference calls to a ThreadPoolExecutor with `max_workers` equal to the number of GPUs. The async event loop SHALL NOT be blocked during inference.

#### Scenario: Concurrent inference does not block event loop

- **WHEN** 4 inference requests are running simultaneously
- **THEN** the FastAPI event loop remains responsive to new incoming HTTP connections

### Requirement: Per-GPU latent cache management

Each GPU SHALL maintain an LRU cache of VAE-encoded latent tensors, keyed by SHA-256 image hash. The cache SHALL have a configurable maximum size (default: 10 entries). When the cache is full and a new entry must be added, the least-recently-used entry SHALL be evicted.

#### Scenario: Cache eviction

- **WHEN** a GPU's latent cache holds 10 entries (the default max) and a new source image arrives
- **THEN** the least-recently-used cached latent is evicted and the new latent is stored

#### Scenario: Cache access updates recency

- **WHEN** a cached latent is reused for a request
- **THEN** that entry is marked as most-recently-used, protecting it from eviction
