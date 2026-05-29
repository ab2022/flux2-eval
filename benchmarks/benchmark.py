#!/usr/bin/env python3
"""
FLUX.2-dev Image Editing Load Test Harness

Sweeps across concurrency levels, collects per-request metrics from both
client and server, and outputs structured JSON results.

Usage:
    # Single-prompt mode
    python benchmark.py \
        --url http://localhost:8000/edit \
        --image test.png \
        --prompt "Change the hat color to red" \
        --concurrency 1,2,4,8 \
        --requests-per-level 20 \
        --output results.json

    # Related-request mode (same image, many prompts)
    python benchmark.py \
        --url http://localhost:8000/edit \
        --image test.png \
        --prompts-file edits.txt \
        --concurrency 1,2,4 \
        --output results.json
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestMetric:
    request_id: str
    concurrency_level: int
    submit_time: float
    response_time: float
    queue_latency_s: float
    service_time_s: float
    vae_encode_time_s: float
    total_latency_s: float
    cache_hit: bool
    gpu_id: int
    status_code: int
    error: str = ""


@dataclass
class LevelSummary:
    concurrency: int
    total_requests: int
    successful: int
    failed: int
    throughput_rps: float
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    latency_mean_s: float
    queue_latency_mean_s: float
    service_time_mean_s: float
    service_time_cache_hit_mean_s: float
    service_time_cache_miss_mean_s: float
    cache_hits: int
    cache_misses: int
    gpu_utilization: dict


# ---------------------------------------------------------------------------
# Request sender
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 600  # seconds


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    image_path: Optional[Path],
    image_bytes: Optional[bytes],
    prompt: str,
    num_steps: int,
    concurrency_level: int,
    request_num: int,
    image_hash: Optional[str] = None,
) -> RequestMetric:
    """Send a single edit request and collect metrics."""
    t_start = time.monotonic()
    try:
        data = aiohttp.FormData()

        if image_bytes is not None and image_hash is None:
            # First request: upload image
            data.add_field(
                "image",
                image_bytes,
                filename=image_path.name if image_path else "image.png",
                content_type="application/octet-stream",
            )
        elif image_hash is not None and image_bytes is not None:
            # Upload image + provide hash
            data.add_field(
                "image",
                image_bytes,
                filename=image_path.name if image_path else "image.png",
                content_type="application/octet-stream",
            )
            data.add_field("image_hash", image_hash)
        elif image_hash is not None:
            # Cache-only: just the hash, no image upload
            data.add_field("image_hash", image_hash)
        else:
            raise ValueError("Either image_bytes or image_hash must be provided")

        data.add_field("prompt", prompt)
        data.add_field("num_steps", str(num_steps))

        async with session.post(url, data=data) as resp:
            t_end = time.monotonic()
            body = await resp.json()
            metrics = body.get("metrics", {})
            return RequestMetric(
                request_id=body.get("request_id", f"req-{request_num}"),
                concurrency_level=concurrency_level,
                submit_time=t_start,
                response_time=t_end,
                queue_latency_s=metrics.get("queue_latency_s", 0),
                service_time_s=metrics.get("service_time_s", 0),
                vae_encode_time_s=metrics.get("vae_encode_time_s", 0),
                total_latency_s=round(t_end - t_start, 3),
                cache_hit=metrics.get("cache_hit", False),
                gpu_id=metrics.get("gpu_id", -1),
                status_code=resp.status,
            )
    except Exception as e:
        t_end = time.monotonic()
        return RequestMetric(
            request_id=f"req-{request_num}",
            concurrency_level=concurrency_level,
            submit_time=t_start,
            response_time=t_end,
            queue_latency_s=0,
            service_time_s=0,
            vae_encode_time_s=0,
            total_latency_s=round(t_end - t_start, 3),
            cache_hit=False,
            gpu_id=-1,
            status_code=0,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Concurrency-level runner
# ---------------------------------------------------------------------------


async def run_level_single_prompt(
    url: str,
    image_path: Path,
    image_bytes: bytes,
    prompt: str,
    num_steps: int,
    concurrency: int,
    num_requests: int,
) -> list[RequestMetric]:
    """Run num_requests of the same prompt at a given concurrency level."""
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestMetric] = []

    async def bounded_request(i: int):
        async with sem:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                m = await send_request(
                    session, url, image_path, image_bytes, prompt,
                    num_steps, concurrency, i,
                )
                results.append(m)

    tasks = [bounded_request(i) for i in range(num_requests)]
    await asyncio.gather(*tasks)
    return results


async def run_level_related_prompts(
    url: str,
    image_path: Path,
    image_bytes: bytes,
    prompts: list[str],
    num_steps: int,
    concurrency: int,
) -> list[RequestMetric]:
    """
    Run related requests: same image, different prompts.
    First request uploads image; subsequent use image_hash for cache reuse.
    """
    results: list[RequestMetric] = []
    image_hash: Optional[str] = None

    if concurrency == 1:
        # Sequential: maximise cache hits
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i, prompt in enumerate(prompts):
                if i == 0:
                    m = await send_request(
                        session, url, image_path, image_bytes, prompt,
                        num_steps, concurrency, i,
                    )
                    if m.status_code == 200:
                        # Extract image_hash from server response for next requests
                        # Re-fetch from the metric (it's in the response)
                        image_hash = getattr(m, '_server_image_hash', None)
                        # We need to get image_hash from the server response.
                        # Since we stored it in the metric's cache_hit field path,
                        # let's re-send with the hash computed client-side.
                        import hashlib
                        image_hash = hashlib.sha256(image_bytes).hexdigest()
                else:
                    m = await send_request(
                        session, url, image_path, None, prompt,
                        num_steps, concurrency, i, image_hash=image_hash,
                    )
                results.append(m)
    else:
        # Concurrent related requests: first uploads, rest use hash
        import hashlib
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(i: int, prompt: str):
            async with sem:
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    if i == 0:
                        m = await send_request(
                            session, url, image_path, image_bytes, prompt,
                            num_steps, concurrency, i,
                        )
                    else:
                        m = await send_request(
                            session, url, image_path, None, prompt,
                            num_steps, concurrency, i,
                            image_hash=image_hash,
                        )
                    results.append(m)

        # Send first request synchronously to prime the cache
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            m = await send_request(
                session, url, image_path, image_bytes, prompts[0],
                num_steps, concurrency, 0,
            )
            results.append(m)

        # Then send remaining concurrently
        remaining_tasks = [
            bounded_request(i, prompts[i]) for i in range(1, len(prompts))
        ]
        await asyncio.gather(*remaining_tasks)

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-1) from a sorted list."""
    if not sorted_values:
        return 0.0
    idx = int(p * (len(sorted_values) - 1))
    return sorted_values[idx]


def summarize(metrics: list[RequestMetric], concurrency: int) -> LevelSummary:
    successful = [m for m in metrics if m.status_code == 200]
    latencies = sorted([m.total_latency_s for m in successful])

    if metrics:
        wall_time = max(m.response_time for m in metrics) - min(
            m.submit_time for m in metrics
        )
    else:
        wall_time = 0

    gpu_counts: dict[int, int] = {}
    for m in successful:
        gpu_counts[m.gpu_id] = gpu_counts.get(m.gpu_id, 0) + 1

    cache_hit_times = [m.service_time_s for m in successful if m.cache_hit]
    cache_miss_times = [m.service_time_s for m in successful if not m.cache_hit]

    return LevelSummary(
        concurrency=concurrency,
        total_requests=len(metrics),
        successful=len(successful),
        failed=len(metrics) - len(successful),
        throughput_rps=round(len(successful) / wall_time, 3) if wall_time > 0 else 0,
        latency_p50_s=round(percentile(latencies, 0.50), 3),
        latency_p95_s=round(percentile(latencies, 0.95), 3),
        latency_p99_s=round(percentile(latencies, 0.99), 3),
        latency_mean_s=round(statistics.mean(latencies), 3) if latencies else 0,
        queue_latency_mean_s=round(
            statistics.mean([m.queue_latency_s for m in successful]), 3
        )
        if successful
        else 0,
        service_time_mean_s=round(
            statistics.mean([m.service_time_s for m in successful]), 3
        )
        if successful
        else 0,
        service_time_cache_hit_mean_s=round(statistics.mean(cache_hit_times), 3)
        if cache_hit_times
        else 0,
        service_time_cache_miss_mean_s=round(statistics.mean(cache_miss_times), 3)
        if cache_miss_times
        else 0,
        cache_hits=len(cache_hit_times),
        cache_misses=len(cache_miss_times),
        gpu_utilization=gpu_counts,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(s: LevelSummary):
    print(f"\n{'=' * 65}")
    print(f"  Concurrency: {s.concurrency}")
    print(f"  Requests:    {s.successful}/{s.total_requests} successful")
    if s.failed > 0:
        print(f"  Failed:      {s.failed}")
    print(f"  Throughput:  {s.throughput_rps:.3f} req/s")
    print(
        f"  Latency:     p50={s.latency_p50_s:.2f}s  "
        f"p95={s.latency_p95_s:.2f}s  p99={s.latency_p99_s:.2f}s  "
        f"mean={s.latency_mean_s:.2f}s"
    )
    print(f"  Queue wait:  {s.queue_latency_mean_s:.3f}s (mean)")
    print(f"  Service:     {s.service_time_mean_s:.2f}s (mean)")
    if s.cache_hits > 0 or s.cache_misses > 0:
        total = s.cache_hits + s.cache_misses
        pct = (s.cache_hits / total * 100) if total > 0 else 0
        print(
            f"  Cache:       {s.cache_hits} hits / {s.cache_misses} misses "
            f"({pct:.0f}% hit rate)"
        )
        if s.service_time_cache_hit_mean_s > 0:
            print(f"    hit  svc:  {s.service_time_cache_hit_mean_s:.2f}s (mean)")
        if s.service_time_cache_miss_mean_s > 0:
            print(f"    miss svc:  {s.service_time_cache_miss_mean_s:.2f}s (mean)")
    print(f"  GPU spread:  {dict(s.gpu_utilization)}")
    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="FLUX.2-dev load testing harness"
    )
    parser.add_argument("--url", default="http://localhost:8000/edit")
    parser.add_argument("--image", required=True, help="Path to source image")
    parser.add_argument("--prompt", default=None, help="Single prompt (for repeat mode)")
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="File with one prompt per line (for related-request mode)",
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    if args.prompt is None and args.prompts_file is None:
        parser.error("Either --prompt or --prompts-file is required")

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    image_bytes = image_path.read_bytes()
    levels = [int(c.strip()) for c in args.concurrency.split(",")]

    # Load prompts
    prompts: Optional[list[str]] = None
    if args.prompts_file:
        pf = Path(args.prompts_file)
        if not pf.exists():
            parser.error(f"Prompts file not found: {pf}")
        prompts = [line.strip() for line in pf.read_text().splitlines() if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {pf}")

    all_results: dict = {}

    for level in levels:
        if prompts:
            print(
                f"\n>>> Related-request mode: {len(prompts)} prompts "
                f"at concurrency={level}..."
            )
            metrics = await run_level_related_prompts(
                args.url, image_path, image_bytes, prompts,
                args.num_steps, level,
            )
        else:
            print(
                f"\n>>> Running {args.requests_per_level} requests "
                f"at concurrency={level}..."
            )
            metrics = await run_level_single_prompt(
                args.url, image_path, image_bytes, args.prompt,
                args.num_steps, level, args.requests_per_level,
            )

        summary = summarize(metrics, level)
        print_summary(summary)
        all_results[str(level)] = {
            "summary": asdict(summary),
            "requests": [asdict(m) for m in metrics],
        }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
