#!/usr/bin/env bash
# setup_moondream.sh — one-shot install of the Moondream2 environment (VQA +
# .detect() grounding) on a fresh machine: system packages, a dedicated
# venv, pinned Python deps, and the model weights.
#
# Captures exactly what was done by hand on the rover's Jetson
# (JetPack R39 / CUDA 13.2 / Orin) on 2026-08-24 — see the
# `moondream_jetson_setup` memory for that session's details/gotchas.
#
# Usage
# ─────
#   ./setup_moondream.sh                              # defaults below
#   MODEL_DIR=/data/models/moondream2 ./setup_moondream.sh
#   VENV_DIR=.venv-md HF_REPO=vikhyatk/moondream2 ./setup_moondream.sh
#
# After it finishes:
#   source .venv-moondream/bin/activate
#   python moondream_cloud_server.py --model-path ~/models/moondream2 --port 8767
#   python test_moondream_detect.py --model-path ~/models/moondream2 --device 0
#   python test_moondream_detect_web.py --model-path ~/models/moondream2 --web-port 5000
#
# Requires: passwordless sudo (for apt) if python3-venv/pip/git-lfs aren't
# already installed, and enough disk for the ~3.6GB model.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv-moondream}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/moondream2}"
HF_REPO="${HF_REPO:-vikhyatk/moondream2}"
REQ_FILE="${REQ_FILE:-requirements-moondream.txt}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

echo "── System packages ──────────────────────────────────────────────"
if ! python3 -c "import venv" >/dev/null 2>&1 || ! command -v pip3 >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-pip python3-venv git-lfs
    else
        echo "No apt-get found — install python3-pip, python3-venv, git-lfs manually" >&2
        exit 1
    fi
fi
python3 --version

echo "── Virtualenv: $VENV_DIR ────────────────────────────────────────"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install virtualenv -q

echo "── Python deps (transformers<5 pinned — see requirements-moondream.txt) ──"
if [ ! -f "$REQ_FILE" ]; then
    echo "Missing $REQ_FILE next to this script" >&2
    exit 1
fi
pip install -r "$REQ_FILE" -q

echo "── Sanity: torch / CUDA ─────────────────────────────────────────"
python3 -c "
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
"

echo "── Model weights: $MODEL_DIR (repo: $HF_REPO) ───────────────────"
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Already present — skipping download"
else
    python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='$HF_REPO', local_dir='$MODEL_DIR')
print('downloaded to', path)
"
fi

echo "── Verifying .detect() is present on the loaded model ──────────"
python3 -c "
import sys
sys.path.insert(0, '.')
from moondream_cloud_server import InferenceEngine
engine = InferenceEngine(model_path='$MODEL_DIR', device_map='auto')
engine.load()
assert hasattr(engine._model, 'detect'), 'model has no .detect() — checkpoint/transformers mismatch'
assert hasattr(engine._model, 'answer_question'), 'model has no .answer_question()'
print('OK — .detect() and .answer_question() both present')
"

echo
echo "Done. Model at $MODEL_DIR, venv at $VENV_DIR."
echo "Next: source $VENV_DIR/bin/activate && python moondream_cloud_server.py --model-path $MODEL_DIR"
