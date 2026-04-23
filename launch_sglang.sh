#!/usr/bin/env bash
# Helper to launch sglang server variants for Phase 0 comparison.
# Each variant serves on port 6006. Kill the previous one before starting the next.
#
# Usage:
#   ./launch_sglang.sh baseline
#   ./launch_sglang.sh eagle       # needs --speculative-draft-model-path set below
#   ./launch_sglang.sh ngram
#   ./launch_sglang.sh standalone  # needs --speculative-draft-model-path set below
#
# Edit MODEL_PATH and DRAFT_MODEL_PATH before use.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/glm-4-9b-chat-int4}"      # adjust to actual path
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"                                    # set for eagle / standalone
HOST="0.0.0.0"
PORT="${PORT:-6006}"
MEM_FRAC="${MEM_FRAC:-0.85}"

VARIANT="${1:-baseline}"

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --host "$HOST" --port "$PORT"
  --mem-fraction-static "$MEM_FRAC"
  --disable-radix-cache       # simpler, more deterministic for Phase 0 bench
  --trust-remote-code
)

case "$VARIANT" in
  baseline)
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}"
    ;;
  ngram)
    # NGRAM needs no draft model. Tune num-draft-tokens / trie-depth later.
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --speculative-algorithm NGRAM \
      --speculative-num-draft-tokens 16 \
      --speculative-ngram-max-trie-depth 18 \
      --disable-overlap-schedule
    ;;
  standalone)
    if [[ -z "$DRAFT_MODEL_PATH" ]]; then
      echo "DRAFT_MODEL_PATH is required for standalone"; exit 2
    fi
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --speculative-algorithm STANDALONE \
      --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
      --speculative-num-steps 3 \
      --speculative-eagle-topk 4 \
      --speculative-num-draft-tokens 8
    ;;
  eagle)
    if [[ -z "$DRAFT_MODEL_PATH" ]]; then
      echo "DRAFT_MODEL_PATH is required for eagle"; exit 2
    fi
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --speculative-algorithm EAGLE \
      --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 4 \
      --speculative-num-draft-tokens 8
    ;;
  eagle3)
    if [[ -z "$DRAFT_MODEL_PATH" ]]; then
      echo "DRAFT_MODEL_PATH is required for eagle3"; exit 2
    fi
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --speculative-algorithm EAGLE3 \
      --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 4 \
      --speculative-num-draft-tokens 8
    ;;
  *)
    echo "usage: $0 {baseline|ngram|standalone|eagle|eagle3}"
    exit 1
    ;;
esac
