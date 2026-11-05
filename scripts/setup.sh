#!/usr/bin/env bash
# One-time setup: Python env for the inference sidecar + model weights (~1.2 GB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || { echo "uv is required: brew install uv"; exit 1; }

echo "==> Python environment (inference/.venv)"
(cd inference && uv sync --frozen)

fetch() {  # repo file dest
  if [[ -s "$3" ]]; then echo "    have $3"; return; fi
  mkdir -p "$(dirname "$3")"
  echo "    downloading $1/$2"
  curl -fL --progress-bar -o "$3.part" "https://huggingface.co/$1/resolve/main/$2" && mv "$3.part" "$3"
}

echo "==> Model weights (Auto-AVSR LRS3 video model + subword LM)"
fetch Amanvir/LRS3_V_WER19.1 model.json models/vsr/model.json
fetch Amanvir/LRS3_V_WER19.1 model.pth  models/vsr/model.pth
fetch Amanvir/lm_en_subword  model.json models/lm/model.json
fetch Amanvir/lm_en_subword  model.pth  models/lm/model.pth

echo "==> Context-correction LLM (Qwen3.5-4B, 4-bit MLX, ~2.5 GB)"
(cd inference && uv run --frozen python -c "from huggingface_hub import snapshot_download as s; s('mlx-community/Qwen3.5-4B-MLX-4bit')")

echo "==> Done. Build the app with ./scripts/build_app.sh"
