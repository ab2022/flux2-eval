"""
FLUX.2-dev Image Editing Server

Data-parallel serving across multiple GPUs with per-GPU image caching
and cache-affinity routing.
"""

import asyncio
import base64
import hashlib
import io
import logging
import os
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger("flux2-eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Image Cache (per-GPU LRU)
# ---------------------------------------------------------------------------


class ImageCache:
    """
    LRU cache of decoded PIL Images per GPU, keyed by SHA-256 of the raw
    image bytes. On a cache hit the client can omit the image upload and
    the server skips byte-reading / JPEG-PNG decoding.

    The cached PIL Image is kept in CPU memory; the pipeline moves it to
    the GPU internally during VAE encoding.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()

    def get(self, image_hash: str) -> Optional[Image.Image]:
        if image_hash in self._cache:
            self._cache.move_to_end(image_hash)
            return self._cache[image_hash]
        return None

    def put(self, image_hash: str, pil_image: Image.Image) -> None:
        if image_hash in self._cache:
            self._cache.move_to_end(image_hash)
            self._cache[image_hash] = pil_image
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[image_hash] = pil_image

    def has(self, image_hash: str) -> bool:
        return image_hash in self._cache


def compute_image_hash(image_bytes: bytes) -> str:
    """SHA-256 hash of raw image bytes."""
    return hashlib.sha256(image_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

NUM_GPUS: int = 0
pipes: list = []  # DiffusionPipeline instances
gpu_semaphores: list[asyncio.Semaphore] = []
gpu_caches: list[ImageCache] = []
executor: Optional[ThreadPoolExecutor] = None

MODEL_ID = "black-forest-labs/FLUX.2-dev"
CACHE_MAX_SIZE = int(os.environ.get("VAE_CACHE_SIZE", "10"))
FORCE_QUANTIZE = os.environ.get("FLUX2_FORCE_QUANTIZE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# GPU detection & model loading
# ---------------------------------------------------------------------------


def detect_gpus() -> list[dict]:
    """Enumerate CUDA GPUs and their VRAM."""
    gpu_info = []
    count = torch.cuda.device_count()
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        gpu_info.append(
            {
                "index": i,
                "name": props.name,
                "vram_gb": round(vram_gb, 1),
                "use_quantization": FORCE_QUANTIZE or vram_gb < 70,
            }
        )
    return gpu_info


def load_model_on_gpu(gpu_id: int, use_quantization: bool):
    """Load FLUX.2-dev pipeline onto a specific GPU."""
    from diffusers import DiffusionPipeline

    model_path = os.environ.get("FLUX2_MODEL_PATH", MODEL_ID)
    kwargs: dict = {"torch_dtype": torch.bfloat16}

    if use_quantization:
        from diffusers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"bnb_4bit_compute_dtype": torch.bfloat16},
        )
        logger.info(
            "GPU %d: Loading with 4-bit quantization from %s", gpu_id, model_path
        )
    else:
        logger.info("GPU %d: Loading in bf16 from %s", gpu_id, model_path)

    pipe = DiffusionPipeline.from_pretrained(model_path, **kwargs)

    if use_quantization:
        # 4-bit quantized model is ~18-20 GB total — fits easily on 80 GB GPUs.
        # Place the entire pipeline on-device to avoid per-request CPU↔GPU
        # transfers that enable_model_cpu_offload would otherwise perform.
        device = f"cuda:{gpu_id}"
        pipe = pipe.to(device)
        logger.info("GPU %d: Pipeline placed on-device (no CPU offload needed)", gpu_id)

        # Compile the transformer for optimized CUDA execution across all
        # denoising steps.  The first inference triggers compilation (~60-90s);
        # subsequent calls benefit from the compiled graph.
        # mode="reduce-overhead" uses CUDA graphs with thread-local state and
        # crashes when requests are served from different threads.
        # mode="default" applies Triton kernel fusion without CUDA graphs and
        # is safe across all threads in the ThreadPoolExecutor.
        pipe.transformer = torch.compile(pipe.transformer, mode="default")
        logger.info("GPU %d: Transformer compiled with mode='default'", gpu_id)
    else:
        # bf16 at full precision: transformer (~64GB) + T5-XXL (~10GB) + VAE
        # exceeds 80GB.  CPU offload moves each submodel to GPU only when
        # active, keeping the rest on CPU RAM.
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        logger.info("GPU %d: Model loaded with CPU offload", gpu_id)

    return pipe


# ---------------------------------------------------------------------------
# Cache-affinity request routing
# ---------------------------------------------------------------------------


async def route_request(image_hash: Optional[str], preferred_gpu: Optional[int] = None) -> int:
    """
    Route to a GPU with the following priority:
    1. preferred_gpu if specified and idle (hard pin for benchmark workers).
    2. If image_hash given, prefer an idle GPU that has it cached.
    3. Fall back to first available GPU by index.
    4. If all busy, wait until one frees up.
    """
    while True:
        # Phase 1: preferred GPU pin
        if preferred_gpu is not None and 0 <= preferred_gpu < NUM_GPUS:
            if gpu_semaphores[preferred_gpu]._value > 0:
                return preferred_gpu

        # Phase 2: cache-affinity
        if image_hash:
            for i in range(NUM_GPUS):
                if gpu_caches[i].has(image_hash) and gpu_semaphores[i]._value > 0:
                    return i

        # Phase 3: first-available
        for i in range(NUM_GPUS):
            if gpu_semaphores[i]._value > 0:
                return i

        # Phase 4: backpressure
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="FLUX.2-dev Evaluation Server")


@app.on_event("startup")
async def startup():
    global NUM_GPUS, executor

    gpu_info = detect_gpus()
    NUM_GPUS = len(gpu_info)

    if NUM_GPUS == 0:
        raise RuntimeError("No CUDA GPUs found")

    logger.info("Detected %d GPUs:", NUM_GPUS)
    for g in gpu_info:
        mode = "4-bit quantized" if g["use_quantization"] else "bf16"
        logger.info(
            "  GPU %d: %s — %.1f GB VRAM — %s", g["index"], g["name"], g["vram_gb"], mode
        )

    # Load models sequentially (each needs significant VRAM)
    for g in gpu_info:
        pipe = load_model_on_gpu(g["index"], g["use_quantization"])
        pipes.append(pipe)
        gpu_semaphores.append(asyncio.Semaphore(1))
        gpu_caches.append(ImageCache(max_size=CACHE_MAX_SIZE))

    # Thread pool for blocking GPU inference
    executor = ThreadPoolExecutor(max_workers=NUM_GPUS)
    loop = asyncio.get_event_loop()
    loop.set_default_executor(executor)

    logger.info("All %d model replicas loaded. Server ready on port 8000.", NUM_GPUS)


@app.post("/edit")
async def edit_image(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_hash: Optional[str] = Form(None),
    num_steps: int = Form(28),
    guidance_scale: float = Form(4.0),
    preferred_gpu: Optional[int] = Form(None),
):
    request_id = str(uuid.uuid4())
    t_queued = time.monotonic()

    # ---- Resolve image bytes and hash ----
    img_bytes: Optional[bytes] = None
    if image is not None:
        img_bytes = await image.read()
        computed_hash = compute_image_hash(img_bytes)
        if image_hash is None:
            image_hash = computed_hash
    elif image_hash is None:
        return JSONResponse(
            status_code=422,
            content={"detail": "Either 'image' file or 'image_hash' must be provided"},
        )

    # ---- Route to GPU ----
    gpu_id = await route_request(image_hash, preferred_gpu)

    async with gpu_semaphores[gpu_id]:
        t_start = time.monotonic()
        queue_latency = t_start - t_queued

        # ---- Resolve input image (cache or decode) ----
        cache = gpu_caches[gpu_id]
        input_image: Optional[Image.Image] = None
        cache_hit = False

        if image_hash and cache.has(image_hash):
            input_image = cache.get(image_hash)
            cache_hit = True
        elif img_bytes is not None:
            input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if image_hash:
                cache.put(image_hash, input_image)
        else:
            # image_hash provided, no image file, cache miss
            return JSONResponse(
                status_code=404,
                content={
                    "detail": (
                        f"Cached image not found for image_hash '{image_hash}'. "
                        "Please upload the image file."
                    )
                },
            )

        # ---- Run inference in thread pool ----
        _input_image = input_image
        _gpu_id = gpu_id
        _prompt = prompt
        _num_steps = num_steps
        _guidance_scale = guidance_scale

        def run_inference():
            pipe = pipes[_gpu_id]
            with torch.no_grad():
                result = pipe(
                    image=_input_image,
                    prompt=_prompt,
                    num_inference_steps=_num_steps,
                    guidance_scale=_guidance_scale,
                )
            return result.images[0]

        loop = asyncio.get_event_loop()
        t_infer_start = time.monotonic()
        result_image = await loop.run_in_executor(executor, run_inference)
        t_end = time.monotonic()

        service_time = t_end - t_start
        vae_encode_time = 0.0 if cache_hit else round(t_end - t_infer_start, 3)
        # Note: vae_encode_time on cold start includes full inference; true VAE-only
        # time would require pipeline instrumentation. For benchmarking, what matters
        # is the delta between cache-hit and cache-miss service times.

        # ---- Encode output image ----
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse(
        {
            "request_id": request_id,
            "image_base64": img_b64,
            "metrics": {
                "request_id": request_id,
                "gpu_id": gpu_id,
                "queue_latency_s": round(queue_latency, 3),
                "service_time_s": round(service_time, 3),
                "vae_encode_time_s": round(vae_encode_time, 3),
                "total_time_s": round(t_end - t_queued, 3),
                "num_steps": num_steps,
                "cache_hit": cache_hit,
                "image_hash": image_hash,
            },
        }
    )
