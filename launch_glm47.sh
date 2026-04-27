#!/usr/bin/env bash
# GLM-4.7-Flash sglang server 启动脚本
# 用法:
#   ./launch_glm47.sh baseline       # 不带投机采样
#   ./launch_glm47.sh nextn          # MTP/NextN 投机采样（用模型自带 nextn 层做 draft）
#   ./launch_glm47.sh ngram          # NGRAM 投机采样（仅做对照）
#
# 环境变量:
#   PORT      监听端口，默认 6006
#   MEM_FRAC  显存预算，默认 0.85
#   STEPS     speculative-num-steps，默认 3
#   DRAFT     speculative-num-draft-tokens，默认 4
set -euo pipefail

VARIANT="${1:-baseline}"
PORT="${PORT:-6006}"
MEM_FRAC="${MEM_FRAC:-0.85}"
STEPS="${STEPS:-3}"
DRAFT="${DRAFT:-4}"

MODEL_PATH=/root/autodl-tmp/models/ZhipuAI/GLM-4.7-Flash
PY=/root/autodl-tmp/conda_envs/sglang/bin/python
LOGDIR=/root/autodl-tmp/logs
mkdir -p "$LOGDIR"

# JIT 编译 / 链接需要 ninja 和 libcudart 在 PATH / LIBRARY_PATH
export PATH=/root/autodl-tmp/conda_envs/sglang/bin:$PATH
export LIBRARY_PATH=/root/autodl-tmp/conda_envs/sglang/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/root/autodl-tmp/conda_envs/sglang/lib:${LD_LIBRARY_PATH:-}

# 先杀掉旧 server (ps + awk 避免 -f 匹配自身导致自杀)
ps -eo pid,args | awk '/launch[_]server/ {print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 3

# 默认开启 cuda graph; Blackwell 若失败可设 NO_CUDA_GRAPH=1
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
    # GLM-4.7-Flash 自带 num_nextn_predict_layers=1 (MTP)
    # sglang 用 EAGLE worker + 同模型路径加载 nextn 层做 draft
    LOG="$LOGDIR/server_nextn.log"
    setsid nohup "$PY" -m sglang.launch_server "${COMMON[@]}" \
      --speculative-algorithm NEXTN \
      --speculative-draft-model-path "$MODEL_PATH" \
      --speculative-num-steps "$STEPS" \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens "$DRAFT" \
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
    echo "用法: $0 {baseline|nextn|ngram}"; exit 1
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
