#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${JUDGE_MODEL:?Set JUDGE_MODEL to a local Qwen3.5-9B checkpoint}"
export CUDA_VISIBLE_DEVICES="${JUDGE_GPU:?Set JUDGE_GPU to an available judge GPU}"
exec "${JUDGE_PYTHON:-python}" examples/ithv/qwen_judge_server.py \
  --model-path "$JUDGE_MODEL" --device cuda:0 --dtype bfloat16 \
  --batch-size 4 --max-new-tokens 256 --temperature 0.7 --top-p 0.8 --top-k 20 \
  --port "${JUDGE_PORT:-8003}" \
  --audit-path "${JUDGE_AUDIT_PATH:-outputs/judge_failures.jsonl}"
