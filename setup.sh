#!/bin/bash
set -e

echo "=== FLUX.2-dev Evaluation Platform Setup ==="

cd /home/azureuser/flux2-eval

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv .venv
fi

source .venv/bin/activate
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.9 support
echo ""
echo "=== Installing PyTorch with CUDA support ==="
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu129

# Install project dependencies
echo ""
echo "=== Installing project dependencies ==="
pip install -e .

# Login to HuggingFace
echo ""
echo "=== HuggingFace Login ==="
if [ -n "$HF_TOKEN" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "Logged in to HuggingFace."
else
    echo "WARNING: HF_TOKEN not set. Set it before downloading the model."
fi

# Verify GPU access
echo ""
echo "=== GPU Verification ==="
python -c "
import torch
n = torch.cuda.device_count()
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {n}')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} — {p.total_mem / 1024**3:.1f} GB VRAM')
"

echo ""
echo "=== Setup Complete ==="
echo "To start the server:  source .venv/bin/activate && python server.py"
echo "To run benchmarks:    source .venv/bin/activate && python benchmark.py --help"
