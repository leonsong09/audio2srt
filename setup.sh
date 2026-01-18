#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODE="${MODE:-faster-whisper}"          # faster-whisper | whisper-api
MODEL_NAME="${MODEL_NAME:-belle-whisper-large-v3-zh-punct}"    # tiny/base/small/medium/large-v2/large-v3/large-v3-turbo/belle-*
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/AppData/models}"
DATASET_DIR="${DATASET_DIR:-/tmp/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/output}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/configs/runtime.env}"

echo "==> Root: $ROOT_DIR"
echo "==> Mode: $MODE"
echo "==> Model: $MODEL_NAME"
echo "==> Model dir: $MODEL_DIR"
echo "==> Dataset dir: $DATASET_DIR"
echo "==> Output dir: $OUTPUT_DIR"

mkdir -p "$MODEL_DIR" "$DATASET_DIR" "$OUTPUT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

if [ "$MODE" = "faster-whisper" ]; then
  pip install "faster-whisper>=1.0.0" "huggingface_hub>=0.24.0"
  MODEL_NAME="$MODEL_NAME" MODEL_DIR="$MODEL_DIR" python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_name = os.environ.get("MODEL_NAME", "belle-whisper-large-v3-zh-punct")
model_dir = Path(os.environ.get("MODEL_DIR", "./AppData/models")).expanduser()
repo_map = {
    "belle-whisper-large-v2-zh": "BELLE-2/Belle-whisper-large-v2-zh",
    "belle-whisper-large-v3-zh": "BELLE-2/Belle-whisper-large-v3-zh",
    "belle-whisper-large-v3-zh-punct": "BELLE-2/Belle-whisper-large-v3-zh-punct",
    "belle-whisper-large-v3-turbo-zh": "BELLE-2/Belle-whisper-large-v3-turbo-zh",
}
repo_id = repo_map.get(model_name, f"Systran/faster-whisper-{model_name}")
target_dir = model_dir / model_name
target_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=str(target_dir),
    local_dir_use_symlinks=False,
)
print(f"Model downloaded: {target_dir}")
PY
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg not found in PATH."
fi

cat > "$ENV_FILE" <<EOF
VIDEO_CAPTIONER_APPDATA=$ROOT_DIR/AppData
VIDEO_CAPTIONER_DATASET=$DATASET_DIR
VIDEO_CAPTIONER_OUTPUT=$OUTPUT_DIR
EOF

echo "==> Done."
echo "Activate env: source $VENV_DIR/bin/activate"
echo "Load paths: source $ENV_FILE"
