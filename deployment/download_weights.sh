#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/deployment/model_weights"
FOLDER_URL="https://drive.google.com/drive/folders/19yJcSJvGmpGlo0X-0owQKrrkPFmPKVt8?usp=sharing"

mkdir -p "${OUTPUT_DIR}"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found on PATH. Activate the lelan conda environment first." >&2
  exit 1
fi

exec gdown --folder -c "${FOLDER_URL}" -O "${OUTPUT_DIR}"
