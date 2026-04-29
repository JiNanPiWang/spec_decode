#!/usr/bin/env bash
# GLM-4.7-Flash sglang server 启动脚本
# 用法:
#   ./launch_glm47.sh baseline        # 不开投机采样（参照基线）
#   ./launch_glm47.sh nextn           # NEXTN 算法 + 自带 MTP 头（chain，topk=1）
#   ./launch_glm47.sh eagle           # EAGLE 算法 + 自带 MTP 头当 draft（chain，topk=1）— 即"EAGLE v1 风格"
#   ./launch_glm47.sh eagle2          # EAGLE 算法 + 自带 MTP 头 + tree 采样（topk=4，可调）— 即"EAGLE2 风格"
#   ./launch_glm47.sh eagle3          # EAGLE3（draft = GLM-4.7-Flash-Eagle3 单层 head，HF 下载）
#   ./launch_glm47.sh standalone      # STANDALONE（draft = GLM-4.5-0.6B-v3 小模型）
#   ./launch_glm47.sh ngram           # NGRAM（无 draft 模型）
#
# 环境变量:
#   PORT          监听端口，默认 6006
#   MEM_FRAC      显存预算，默认 0.85
#   STEPS         speculative-num-steps，默认 3
#   DRAFT         speculative-num-draft-tokens，默认 4
#   TOPK          speculative-eagle-topk，默认 1（chain）；EAGLE2/3 想开 tree 设 4 或 8
#   DRAFT_PATH    显式覆盖 draft 模型路径
#   NO_CUDA_GRAPH 设 1 关闭 cuda graph（Blackwell 出错时用）
set -euo pipefail

VARIANT="${1:-baseline}"
PORT="${PORT:-6006}"
MEM_FRAC="${MEM_FRAC:-0.85}"
STEPS="${STEPS:-3}"
DRAFT="${DRAFT:-4}"
TOPK="${TOPK:-1}"

MODEL_PATH=/root/autodl-tmp/models/ZhipuAI/GLM-4.7-Flash
EAGLE3_DRAFT=/root/autodl-tmp/models/GLM-4.7-Flash-Eagle3
STANDALONE_DRAFT=/root/autodl-tmp/models/GLM-4.5-0.6B-v3
PY=/root/autodl-tmp/conda_envs/sglang/bin/python
LOGDIR=/root/autodl-tmp/logs
mkdir -p "$LOGDIR"

# JIT 编译 / 链接需要 ninja 与 libcudart 在 PATH / LIBRARY_PATH
export PATH=/root/autodl-tmp/conda_envs/sglang/bin:$PATH
export LIBRARY_PATH=/root/autodl-tmp/conda_envs/sglang/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/root/autodl-tmp/conda_envs/sglang/lib:${LD_LIBRARY_PATH:-}
# 允许 draft 的 max_position_embeddings (4096) 小于 target (202752)
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# 杀残留 server（ps + awk 方括号防自匹配）
ps -eo pid,args | awk '/launch[_]server/ {print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 3

NO_CUDA_GRAPH="${NO_CUDA_GRAPH:-0}"
COMMON=(
  --model-path "$MODEL_PATH"
  --host 0.0.0.0 --port "$PORT"
  --mem-fraction-static "$MEM_FRAC"
  --trust-remote-code
)
if [ "$NO_CUDA_GRAPH" = "1" ]; then
  COMMON+=(--disable-cuda-graph --disable-piecewise-cuda-graph)
fi

case "$VARIANT" in
  baseline)
    LOG="$LOGDIR/server_baseline.log"
    setsid nohup "$PY" -m sglang.launch_server "${COMMON[@]}" \
      > "$LOG" 2>&1 < /dev/null &
    ;;
  nextn)
    # GLM-4.7-Flash 自带 num_nextn_predict_layers=1（MTP）
    LOG="$LOGDIR/server_nextn.log"
    setsid nohup "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm NEXTN \
      --speculative-draft-model-path "$MODEL_PATH" \
      --speculative-num-steps "$STEPS" \
      --speculative-eagle-topk "$TOPK" \
      --speculative-num-draft-tokens "$DRAFT" \
      > "$LOG" 2>&1 < /dev/null &
    ;;
  eagle3)
    # EAGLE3 head：单层 LlamaForCausalLMEagle3，hidden 2048
    DRAFT_MODEL="${DRAFT_PATH:-$EAGLE3_DRAFT}"
    LOG="$LOGDIR/server_eagle3.log"
    setsid nohup "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm EAGLE3 \
      --speculative-draft-model-path "$DRAFT_MODEL" \
      --speculative-num-steps "$STEPS" \
      --speculative-eagle-topk "$TOPK" \
      --speculative-num-draft-tokens "$DRAFT" \
      > "$LOG" 2>&1 < /dev/null &
    ;;
  standalone)
    # 独立小模型 draft（Qwen2-0.6B）；要点：draft 权重未量化
    DRAFT_MODEL="${DRAFT_PATH:-$STANDALONE_DRAFT}"
    LOG="$LOGDIR/server_standalone.log"
    setsid nohup "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm STANDALONE \
      --speculative-draft-model-path "$DRAFT_MODEL" \
      --speculative-draft-model-quantization unquant \
      --speculative-num-steps "$STEPS" \
      --speculative-eagle-topk "$TOPK" \
      --speculative-num-draft-tokens "$DRAFT" \
      > "$LOG" 2>&1 < /dev/null &
    ;;
  eagle)
    # EAGLE 算法 + 自带 MTP 头当 draft（official HF README 写法）
    # topk=1 (默认) → 原版 EAGLE chain；如需 EAGLE2 树采样请用 eagle2 variant
    DRAFT_MODEL="${DRAFT_PATH:-$MODEL_PATH}"
    LOG="$LOGDIR/server_eagle.log"
    setsid nohup "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm EAGLE \
      --speculative-draft-model-path "$DRAFT_MODEL" \
      --speculative-num-steps "$STEPS" \
      --speculative-eagle-topk "$TOPK" \
      --speculative-num-draft-tokens "$DRAFT" \
      > "$LOG" 2>&1 < /dev/null &
    ;;
  eagle2)
    # EAGLE2 风格：EAGLE 算法 + tree 采样（topk>1），draft 用自带 MTP 头
    # 默认 topk=4 num_draft_tokens=8；可用 TOPK / DRAFT 覆盖
    DRAFT_MODEL="${DRAFT_PATH:-$MODEL_PATH}"
    E2_TOPK="${TOPK:-4}"
    if [ "$TOPK" = "1" ]; then E2_TOPK=4; fi   # eagle2 默认强制 tree
    E2_DRAFT="$DRAFT"
    if [ "$DRAFT" = "4" ]; then E2_DRAFT=8; fi
    LOG="$LOGDIR/server_eagle2.log"
    setsid nohup "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm EAGLE \
      --speculative-draft-model-path "$DRAFT_MODEL" \
      --speculative-num-steps "$STEPS" \
      --speculative-eagle-topk "$E2_TOPK" \
      --speculative-num-draft-tokens "$E2_DRAFT" \
      > "$LOG" 2>&1 < /dev/null &
    ;;
  ngram)
    LOG="$LOGDIR/server_ngram.log"
    setsid nohup "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm NGRAM \
      --speculative-num-steps 5 \
      --speculative-num-draft-tokens 8 \
      --speculative-ngram-max-bfs-breadth 10 \
      --speculative-ngram-match-type BFS \
      --speculative-ngram-max-trie-depth 18 \
      > "$LOG" 2>&1 < /dev/null &
    ;;
  *)
    echo "用法: $0 {baseline|nextn|eagle|eagle2|eagle3|standalone|ngram}"; exit 1
    ;;
esac

PID=$!
echo "启动 $VARIANT, PID=$PID, 日志=$LOG"
echo "等待 server ready (超时 10 分钟) ..."
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
         "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "READY at t=$((i*10))s"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "进程已死，最后日志:"
    tail -20 "$LOG"
    exit 1
  fi
  sleep 10
done
echo "超时未就绪，最后日志:"
tail -20 "$LOG"
exit 1
