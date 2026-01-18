#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://gitee.com/leons9/audio2srt.git}"
TARGET_DIR="${TARGET_DIR:-/tmp/code/audio2srt}"

if ! command -v git >/dev/null 2>&1; then
  apt-get update && apt-get install -y git
fi

if [ -d "$TARGET_DIR/.git" ]; then
  git -C "$TARGET_DIR" fetch --all --prune
  git -C "$TARGET_DIR" checkout main
  git -C "$TARGET_DIR" pull --ff-only
else
  mkdir -p "$(dirname "$TARGET_DIR")"
  git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
fi

echo "Done: $TARGET_DIR"
