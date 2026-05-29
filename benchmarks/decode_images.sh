#!/bin/bash
#
# Decode all base64 images from serial benchmark JSON results into PNG files.
#
# Usage: bash decode_images.sh
#

set -euo pipefail

RESULTS_DIR="$(dirname "$0")/results"
IMAGES_DIR="$(dirname "$0")/decoded_images"

mkdir -p "$IMAGES_DIR"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    echo "Run run_serial.sh first."
    exit 1
fi

COUNT=0
for JSON_FILE in "$RESULTS_DIR"/*.json; do
    [[ -f "$JSON_FILE" ]] || continue

    BASENAME=$(basename "$JSON_FILE" .json)
    OUT_PNG="$IMAGES_DIR/${BASENAME}.png"

    python3 -c "
import json, base64, sys
with open('$JSON_FILE') as f:
    data = json.load(f)
with open('$OUT_PNG', 'wb') as f:
    f.write(base64.b64decode(data['image_base64']))
"

    COUNT=$((COUNT + 1))
    echo "Decoded: $OUT_PNG"
done

echo ""
echo "Done. $COUNT images saved to $IMAGES_DIR/"
