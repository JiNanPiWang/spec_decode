#!/usr/bin/env bash
# sglang server 启动脚本，用于 Phase 0 几种变体的性能对比。
# 所有变体都在同一个端口上（默认 6006），切换前先 kill 掉前一个。
#
# 用法:
#   ./launch_sglang.sh baseline      # 无投机采样
#   ./launch_sglang.sh ngram         # NGRAM 投机，不需要 draft 模型
#   ./launch_sglang.sh standalone    # 独立小 draft 模型
#   ./launch_sglang.sh eagle         # EAGLE 投机（需要配套 draft 权重）
#   ./launch_sglang.sh eagle3        # EAGLE-3 投机（需要配套 draft 权重）
#
# 环境变量可以覆盖默认路径：
#   MODEL_PATH       target 模型路径
#   DRAFT_MODEL_PATH draft 模型路径（standalone / eagle 场景需要）
#   PORT             监听端口（默认 6006）
#   MEM_FRAC         显存分配比例（默认 0.88）

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/glm-4-32b-0414-gptq-int4}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/autodl-tmp/models/GLM-4.5-0.6B-v3}"
HOST="0.0.0.0"
PORT="${PORT:-6006}"
MEM_FRAC="${MEM_FRAC:-0.88}"

VARIANT="${1:-baseline}"

# 所有变体共享的基础参数（Blackwell 5090 上 cuda graph 有问题，全部禁用）
COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --host "$HOST" --port "$PORT"
  --mem-fraction-static "$MEM_FRAC"
  --quantization gptq_marlin
  --trust-remote-code
  --disable-cuda-graph
  --disable-piecewise-cuda-graph
)

case "$VARIANT" in
  baseline)
    # 不开投机采样，作为对照基线
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}"
    ;;
  ngram)
    # NGRAM 不需要 draft 模型；这组参数来自 2026-04-22 跑出 5x 的配置
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --speculative-algorithm NGRAM \
      --speculative-num-steps 5 \
      --speculative-num-draft-tokens 8 \
      --speculative-ngram-max-bfs-breadth 10 \
      --speculative-ngram-match-type BFS \
      --speculative-ngram-max-trie-depth 18
    ;;
  standalone)
    # 独立 draft 模型；draft 权重全精度，target 走 gptq_marlin
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --speculative-algorithm STANDALONE \
      --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
      --speculative-draft-model-quantization unquant \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 1
    ;;
  eagle)
    # EAGLE 投机（需要配套 draft 权重，GLM-4 目前没有公开权重）
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --speculative-algorithm EAGLE \
      --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 4 \
      --speculative-num-draft-tokens 8
    ;;
  eagle3)
    # EAGLE-3 投机（需要配套 draft 权重）
    exec python3 -m sglang.launch_server "${COMMON_ARGS[@]}" \
      --speculative-algorithm EAGLE3 \
      --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 4 \
      --speculative-num-draft-tokens 8
    ;;
  *)
    echo "用法: $0 {baseline|ngram|standalone|eagle|eagle3}"
    exit 1
    ;;
esac
