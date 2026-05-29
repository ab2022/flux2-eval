#!/bin/bash
#
# Serial benchmark: issue all prompts from sample_prompts.txt against a single
# source image, routing every request to the same GPU via image_hash affinity.
#
# Usage: bash run_serial.sh
#

set -euo pipefail

SERVER_URL="http://localhost:8000/edit"
IMAGE_PATH="$HOME/flux2-eval/src_images/NY_hat_1.png"
PROMPTS_FILE="$HOME/flux2-eval/sample_prompts.txt"
OUTPUT_DIR="$(dirname "$0")/results"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "ERROR: Source image not found: $IMAGE_PATH"
    exit 1
fi

if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "ERROR: Prompts file not found: $PROMPTS_FILE"
    exit 1
fi

# Read prompts into array
mapfile -t PROMPTS < "$PROMPTS_FILE"
TOTAL=${#PROMPTS[@]}

echo "=== Serial Benchmark ==="
echo "Image:   $IMAGE_PATH"
echo "Prompts: $TOTAL from $PROMPTS_FILE"
echo "Output:  $OUTPUT_DIR"
echo ""

IMAGE_HASH=""
TIMED_TOTAL=0
REQUEST_NUM=0

for PROMPT in "${PROMPTS[@]}"; do
    REQUEST_NUM=$((REQUEST_NUM + 1))

    # Sanitize prompt into a filename-safe slug
    SLUG=$(echo "$PROMPT" | sed "s/[^a-zA-Z0-9]/_/g" | sed "s/__*/_/g" | sed "s/^_//;s/_$//")
    OUTFILE="$OUTPUT_DIR/${REQUEST_NUM}_${SLUG}.json"

    echo "[$REQUEST_NUM/$TOTAL] $PROMPT"

    START=$(date +%s%N)

    if [[ $REQUEST_NUM -eq 1 ]]; then
        # First request: upload the image file
        HTTP_CODE=$(curl -s -w "%{http_code}" -o "$OUTFILE" \
            -X POST "$SERVER_URL" \
            -F "image=@${IMAGE_PATH}" \
            -F "prompt=${PROMPT}" \
            -F "num_steps=28" \
            -F "guidance_scale=4.0")

        # Extract image_hash for all subsequent requests
        IMAGE_HASH=$(python3 -c "import json; print(json.load(open('$OUTFILE'))['metrics']['image_hash'])" 2>/dev/null || true)
        if [[ -z "$IMAGE_HASH" ]]; then
            echo "  ERROR: Failed to extract image_hash from response (HTTP $HTTP_CODE)"
            cat "$OUTFILE"
            exit 1
        fi
    else
        # Subsequent requests: pass image_hash only, skip upload
        HTTP_CODE=$(curl -s -w "%{http_code}" -o "$OUTFILE" \
            -X POST "$SERVER_URL" \
            -F "image_hash=${IMAGE_HASH}" \
            -F "prompt=${PROMPT}" \
            -F "num_steps=28" \
            -F "guidance_scale=4.0")
    fi

    END=$(date +%s%N)
    ELAPSED_MS=$(( (END - START) / 1000000 ))
    ELAPSED_S=$(python3 -c "print(f'{$ELAPSED_MS / 1000:.2f}')")

    # Extract server-side metrics
    GPU_ID=$(python3 -c "import json; print(json.load(open('$OUTFILE'))['metrics']['gpu_id'])" 2>/dev/null || echo "?")
    CACHE_HIT=$(python3 -c "import json; print(json.load(open('$OUTFILE'))['metrics']['cache_hit'])" 2>/dev/null || echo "?")
    SVC_TIME=$(python3 -c "import json; print(json.load(open('$OUTFILE'))['metrics']['service_time_s'])" 2>/dev/null || echo "?")

    echo "  HTTP $HTTP_CODE | ${ELAPSED_S}s client | ${SVC_TIME}s server | GPU $GPU_ID | cache_hit=$CACHE_HIT"

    # Accumulate timed total (skip request 1)
    if [[ $REQUEST_NUM -gt 1 ]]; then
        TIMED_TOTAL=$((TIMED_TOTAL + ELAPSED_MS))
    fi
done

TIMED_TOTAL_S=$(python3 -c "print(f'{$TIMED_TOTAL / 1000:.2f}')")

echo ""
echo "=== Summary ==="
echo "Total requests:              $TOTAL"
echo "First request (excluded):    ${PROMPTS[0]}"
echo "Timed requests (2-$TOTAL):     $((TOTAL - 1))"
echo "Total time (requests 2-$TOTAL): ${TIMED_TOTAL_S}s"
echo "Avg time per timed request:  $(python3 -c "print(f'{$TIMED_TOTAL / 1000 / ($TOTAL - 1):.2f}')")s"
echo "Results saved to:            $OUTPUT_DIR/"
