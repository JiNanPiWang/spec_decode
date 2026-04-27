#!/usr/bin/env bash
# 在指定 sglang server 上扫一组并发度，调用 sglang.bench_serving
# 输出每个并发点的 JSON 结果到 outputs/glm47_concurrency/<tag>/c<N>.json
#
# 用法:
#   ./bench_concurrency.sh <tag>
# 例如:
#   ./bench_concurrency.sh baseline
#   ./bench_concurrency.sh nextn
#
# 环境变量:
#   PORT       目标 server 端口，默认 6006
#   IN_LEN     输入长度，默认 1024
#   OUT_LEN    输出长度，默认 512
#   CONC_LIST  并发度列表（空格分隔），默认 "1 4 16 32 64"
set -euo pipefail

TAG="${1:?用法: $0 <tag>}"
PORT="${PORT:-6006}"
IN_LEN="${IN_LEN:-1024}"
OUT_LEN="${OUT_LEN:-256}"
CONC_LIST="${CONC_LIST:-1 8 32 64}"
DATASET="${DATASET:-custom}"        # custom | random-ids | generated-shared-prefix
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/spec_decode/data/prompts_realistic.jsonl}"
MODEL=/root/autodl-tmp/models/ZhipuAI/GLM-4.7-Flash
PY=/root/autodl-tmp/conda_envs/sglang/bin/python

OUTDIR="/root/autodl-tmp/spec_decode/outputs/glm47_concurrency/$TAG"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
: > "$SUMMARY"
echo -e "concurrency\tnum_prompts\tcompleted\tttft_p50_ms\ttpot_p50_ms\titl_p50_ms\toutput_throughput_tok_s\trequest_throughput_req_s\tduration_s" >> "$SUMMARY"

for C in $CONC_LIST; do
  # 请求总数: 至少 64，且不少于并发度 * 8 (保证稳态)
  N=$(( C * 8 ))
  if [ $N -lt 64 ]; then N=64; fi

  OUT="$OUTDIR/c${C}.json"
  echo ""
  echo "============================================="
  echo " [$TAG] 并发=$C, num_prompts=$N, dataset=$DATASET, out=$OUT_LEN"
  echo "============================================="

  COMMON_ARGS=(
    --backend sglang-oai
    --base-url "http://127.0.0.1:$PORT"
    --model "$MODEL"
    --tokenizer "$MODEL"
    --num-prompts "$N"
    --max-concurrency "$C"
    --warmup-requests 2
    --output-file "$OUT"
    --disable-tqdm
  )
  case "$DATASET" in
    custom)
      "$PY" -m sglang.bench_serving "${COMMON_ARGS[@]}" \
        --dataset-name custom \
        --dataset-path "$DATASET_PATH" \
        --sharegpt-output-len "$OUT_LEN" \
        --apply-chat-template 2>&1 | tail -50
      ;;
    random-ids)
      "$PY" -m sglang.bench_serving "${COMMON_ARGS[@]}" \
        --dataset-name random-ids \
        --random-input-len "$IN_LEN" \
        --random-output-len "$OUT_LEN" \
        --random-range-ratio 0.9 2>&1 | tail -50
      ;;
    generated-shared-prefix)
      "$PY" -m sglang.bench_serving "${COMMON_ARGS[@]}" \
        --dataset-name generated-shared-prefix \
        --gsp-num-groups 8 \
        --gsp-prompts-per-group 8 \
        --gsp-system-prompt-len 512 \
        --gsp-question-len 128 \
        --gsp-output-len "$OUT_LEN" 2>&1 | tail -50
      ;;
  esac

  # 解析关键指标，追加到 summary
  if [ -f "$OUT" ]; then
    "$PY" - "$OUT" "$C" "$N" >> "$SUMMARY" <<'PY'
import json, sys
path, c, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path) as f:
    # bench_serving 输出 jsonl, 取最后一行
    lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1])
def g(k, default=0.0):
    v = d.get(k, default)
    return v if v is not None else default
print(f"{c}\t{n}\t{int(g('completed'))}"
      f"\t{g('median_ttft_ms'):.1f}"
      f"\t{g('median_tpot_ms'):.2f}"
      f"\t{g('median_itl_ms'):.2f}"
      f"\t{g('output_throughput'):.2f}"
      f"\t{g('request_throughput'):.3f}"
      f"\t{g('duration'):.1f}")
PY
  fi
done

echo ""
echo "==== summary [$TAG] ===="
cat "$SUMMARY"
