#!/usr/bin/env bash
# setup_omnivla.sh — Install OmniVLA Full inference environment
#
# Usage:
#   bash setup_omnivla.sh
#
# What this does:
#   1. Creates a conda env 'omnivla' with Python 3.10
#   2. Installs PyTorch 2.2.0 (CUDA 11.8 — edit if your CUDA differs)
#   3. Clones and installs the OmniVLA package
#   4. Installs Flash Attention 2 for faster inference
#   5. Downloads the fine-tuned model weights (omnivla-finetuned-cast)
#   6. Installs rover-agent dependencies into the same env
#
# To check your CUDA version before running:
#   nvidia-smi
#
# To change the PyTorch CUDA variant, edit the TORCH_INSTALL line below.
# Common variants:
#   CUDA 11.8 → cu118  (default below)
#   CUDA 12.1 → cu121
#   CUDA 12.4 → cu124

set -euo pipefail

CONDA_ENV="omnivla"
WEIGHTS_DIR="${HOME}/weights"
OMNIVLA_REPO="${HOME}/OmniVLA"
ROVER_AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }

# ── 1. Conda env ──────────────────────────────────────────────────────────────
info "Creating conda env '${CONDA_ENV}' (Python 3.10)..."
if conda env list | grep -q "^${CONDA_ENV} "; then
    warn "Env '${CONDA_ENV}' already exists — skipping creation"
else
    conda create -n "${CONDA_ENV}" python=3.10 -y
fi

# Activate inside script
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
info "Installing PyTorch 2.2.0 (CUDA 11.8)..."
pip install --quiet \
    numpy==1.26.4 \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 3. OmniVLA package ────────────────────────────────────────────────────────
if [ -d "${OMNIVLA_REPO}" ]; then
    info "OmniVLA repo found at ${OMNIVLA_REPO} — pulling latest..."
    git -C "${OMNIVLA_REPO}" pull
else
    info "Cloning OmniVLA..."
    git clone https://github.com/NHirose/OmniVLA.git "${OMNIVLA_REPO}"
fi

info "Installing OmniVLA package..."
pip install --quiet -e "${OMNIVLA_REPO}"

# ── 4. Flash Attention 2 ──────────────────────────────────────────────────────
info "Installing Flash Attention 2 (this may take a few minutes)..."
pip install --quiet packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation

# ── 5. Model weights ──────────────────────────────────────────────────────────
mkdir -p "${WEIGHTS_DIR}"

download_weights() {
    local name="$1"
    local dest="${WEIGHTS_DIR}/${name}"
    if [ -d "${dest}" ]; then
        warn "Weights '${name}' already exist at ${dest} — skipping"
    else
        info "Downloading ${name}..."
        git clone "https://huggingface.co/NHirose/${name}" "${dest}"
    fi
}

# Fine-tuned weights — best outdoor generalisation
download_weights "omnivla-finetuned-cast"

# Edge variant — lighter and faster
download_weights "omnivla-edge"

# ── 6. Rover-agent dependencies ───────────────────────────────────────────────
if [ -f "${ROVER_AGENT_DIR}/requirements.txt" ]; then
    info "Installing rover-agent requirements..."
    pip install --quiet -r "${ROVER_AGENT_DIR}/requirements.txt"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Installation complete!"
echo ""
echo "  Weights location : ${WEIGHTS_DIR}"
echo ""
echo "  To run the inference server:"
echo "    conda activate ${CONDA_ENV}"
echo "    cd ${ROVER_AGENT_DIR}"
echo "    python omnivla_server.py --weights ${WEIGHTS_DIR}/omnivla-finetuned-cast --port 5100"
echo ""
echo "  To run in the background (survives SSH disconnect):"
echo "    tmux new -s omnivla"
echo "    conda activate ${CONDA_ENV} && python omnivla_server.py --weights ${WEIGHTS_DIR}/omnivla-finetuned-cast --port 5100"
echo "    # Ctrl+B then D to detach"
echo ""
