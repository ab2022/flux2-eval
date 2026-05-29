#!/usr/bin/env python3
"""
Parallel benchmark: 40 prompts across 4 GPUs (10 per GPU).

Phase 1 (warmup):  Sequentially warm up each GPU one at a time using
                   preferred_gpu pinning.  This ensures torch.compile
                   traces on each GPU before the benchmark starts, and
                   that each GPU has the source image cached.

Phase 2 (bench):   4 threads run 10 requests each in parallel, each
                   pinned to a specific GPU via preferred_gpu.  Within
                   each thread requests are serial so the GPU is never
                   contended by more than one worker.

Usage: python3 run_bench.py
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SERVER_URL  = "http://localhost:8000/edit"
IMAGE_PATH  = os.path.expanduser("~/flux2-eval/src_images/NY_hat_1.png")
PROMPTS_FILE = os.path.expanduser("~/flux2-eval/sample_prompts.txt")
SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR / "results"
NUM_GPUS    = 4
PROMPTS_PER_GPU = 10
NUM_STEPS   = 2
GUIDANCE_SCALE = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def send_request(prompt, outfile, image_path=None, image_hash=None,
                 preferred_gpu=None):
    """Send a single /edit request via curl. Returns (http_code, elapsed_s, metrics)."""
    cmd = [
        "curl", "-s", "-w", "%{http_code}", "-o", str(outfile),
        "-X", "POST", SERVER_URL,
        "-F", f"prompt={prompt}",
        "-F", f"num_steps={NUM_STEPS}",
        "-F", f"guidance_scale={GUIDANCE_SCALE}",
    ]
    if image_path:
        cmd += ["-F", f"image=@{image_path}"]
    if image_hash:
        cmd += ["-F", f"image_hash={image_hash}"]
    if preferred_gpu is not None:
        cmd += ["-F", f"preferred_gpu={preferred_gpu}"]

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    http_code = result.stdout.strip()
    metrics = {}
    try:
        with open(outfile) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
    except Exception:
        pass

    return http_code, elapsed, metrics


def run_worker(worker_id, prompts, image_hash):
    """Run PROMPTS_PER_GPU requests serially, pinned to worker_id GPU."""
    results = []
    for i, prompt in enumerate(prompts):
        req_num = i + 1
        slug = "".join(c if c.isalnum() else "_" for c in prompt)
        slug = "_".join(p for p in slug.split("_") if p)[:80]
        outfile = OUTPUT_DIR / f"gpu{worker_id}_{req_num}_{slug}.json"

        http_code, elapsed, metrics = send_request(
            prompt, outfile,
            image_hash=image_hash,
            preferred_gpu=worker_id,
        )

        gpu_id    = metrics.get("gpu_id", "?")
        cache_hit = metrics.get("cache_hit", "?")
        svc_time  = metrics.get("service_time_s", "?")

        results.append({
            "worker":    worker_id,
            "req_num":   req_num,
            "prompt":    prompt,
            "http_code": http_code,
            "client_s":  round(elapsed, 3),
            "server_s":  svc_time,
            "gpu_id":    gpu_id,
            "cache_hit": cache_hit,
        })

        print(
            f"  [W{worker_id}] {req_num}/{len(prompts)} | "
            f"{elapsed:.2f}s | GPU {gpu_id} | cache={cache_hit} | "
            f"HTTP {http_code} | {prompt}"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.isfile(IMAGE_PATH):
        print(f"ERROR: Source image not found: {IMAGE_PATH}")
        sys.exit(1)
    if not os.path.isfile(PROMPTS_FILE):
        print(f"ERROR: Prompts file not found: {PROMPTS_FILE}")
        sys.exit(1)

    with open(PROMPTS_FILE) as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    expected = NUM_GPUS * PROMPTS_PER_GPU
    if len(all_prompts) < expected:
        print(f"ERROR: Need {expected} prompts but only found {len(all_prompts)}")
        sys.exit(1)

    prompts = all_prompts[:expected]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Parallel Benchmark ({NUM_GPUS} GPUs x {PROMPTS_PER_GPU} prompts) ===")
    print(f"Image:      {IMAGE_PATH}")
    print(f"Prompts:    {expected} from {PROMPTS_FILE}")
    print(f"Num steps:  {NUM_STEPS}")
    print(f"Output:     {OUTPUT_DIR}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Sequential warmup — one GPU at a time
    #
    # Uses preferred_gpu to pin each warmup request to a specific GPU.
    # This triggers torch.compile tracing per-GPU in a controlled order
    # and seeds each GPU's image cache before the benchmark starts.
    # ------------------------------------------------------------------
    print("--- Phase 1: Warming up all GPUs (sequential, pinned) ---")
    print()

    image_hash = None
    for gpu_idx in range(NUM_GPUS):
        outfile = OUTPUT_DIR / f"warmup_gpu{gpu_idx}.json"
        http_code, elapsed, metrics = send_request(
            f"warmup GPU {gpu_idx}", outfile,
            image_path=IMAGE_PATH,
            image_hash=image_hash,      # None on first call, reused after
            preferred_gpu=gpu_idx,
        )
        gpu_id    = metrics.get("gpu_id", "?")
        cache_hit = metrics.get("cache_hit", "?")

        if image_hash is None:
            image_hash = metrics.get("image_hash")
            if not image_hash:
                print(f"  ERROR: no image_hash in warmup response (HTTP {http_code})")
                try:
                    print(f"  Body: {open(outfile).read()[:400]}")
                except Exception:
                    pass
                sys.exit(1)

        print(f"  GPU {gpu_id}: {elapsed:.2f}s | cache={cache_hit} | HTTP {http_code}")

    print()
    print(f"  image_hash: {image_hash}")
    print(f"  All {NUM_GPUS} GPUs warmed up.")
    print()

    # ------------------------------------------------------------------
    # Phase 2: Benchmark — 4 parallel workers, each pinned to one GPU
    # ------------------------------------------------------------------
    print(f"--- Phase 2: Benchmark ({NUM_GPUS} parallel x {PROMPTS_PER_GPU} serial, pinned) ---")
    print()

    chunks = [
        prompts[i * PROMPTS_PER_GPU : (i + 1) * PROMPTS_PER_GPU]
        for i in range(NUM_GPUS)
    ]

    wall_start = time.monotonic()

    all_results = []
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as pool:
        futures = {
            pool.submit(run_worker, w, chunks[w], image_hash): w
            for w in range(NUM_GPUS)
        }
        for future in as_completed(futures):
            w = futures[future]
            try:
                all_results.extend(future.result())
            except Exception as e:
                print(f"  [W{w}] FAILED: {e}")

    wall_elapsed = time.monotonic() - wall_start

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  Per-GPU Summary")
    print("=" * 72)
    print()
    hdr = f"{'Worker':<8} {'GPU':>4} {'Reqs':>5} {'Total(s)':>9} {'Avg(s)':>8} {'Hits':>5}"
    print(hdr)
    print("-" * len(hdr))

    for w in range(NUM_GPUS):
        wr = [r for r in all_results if r["worker"] == w]
        if not wr:
            print(f"W{w:<7} {'?':>4} {'FAIL':>5} {'-':>9} {'-':>8} {'-':>5}")
            continue
        gpus  = sorted(set(str(r["gpu_id"]) for r in wr))
        total = sum(r["client_s"] for r in wr)
        avg   = total / len(wr)
        hits  = sum(1 for r in wr if r["cache_hit"] is True)
        print(f"W{w:<7} {','.join(gpus):>4} {len(wr):>5} {total:>9.2f} {avg:>8.2f} {hits:>5}")

    print()
    print("=" * 72)
    print("  Overall Summary")
    print("=" * 72)
    print()

    total_reqs   = len(all_results)
    gpus_used    = sorted(set(str(r["gpu_id"]) for r in all_results))
    total_hits   = sum(1 for r in all_results if r["cache_hit"] is True)
    total_client = sum(r["client_s"] for r in all_results)
    throughput   = total_reqs / wall_elapsed if wall_elapsed > 0 else 0
    errors       = sum(1 for r in all_results if r["http_code"] != "200")

    print(f"Total requests:    {total_reqs}")
    print(f"Errors (non-200):  {errors}")
    print(f"GPUs used:         {len(gpus_used)} ({', '.join(gpus_used)})")
    print(f"Cache hits:        {total_hits}/{total_reqs}")
    print(f"Wall-clock time:   {wall_elapsed:.2f}s")
    print(f"Throughput:        {throughput:.2f} req/s")
    print(f"Avg client time:   {total_client / total_reqs:.2f}s/req")
    print(f"Results saved to:  {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
