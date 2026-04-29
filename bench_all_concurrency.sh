#!/usr/bin/env bash
# GLM-4.7-Flash 高并发全 matrix bench
# 默认依次跑 baseline / nextn / eagle2 / eagle3，每个 variant 在 c=1/8/32/64 各跑一次。
# 结果落盘：
#   outputs/glm47_concurrency/<variant>/c{C}.json     (sglang.bench_serving 原始结果)
#   outputs/glm47_concurrency/<variant>/summary.tsv   (单 variant 多并发汇总)
#   outputs/glm47_concurrency/_combined_summary.tsv   (全 variant 全并发汇总，带 accept_length)
#   /root/autodl-tmp/logs/bench_all_overall.log       (总 log)
#   /root/autodl-tmp/logs/bench_all_launch_<v>.log    (每个 variant 的 launch log)
#   /root/autodl-tmp/logs/bench_all_bench_<v>.log     (每个 variant 的 bench log)
#
# 用法：
#   bash bench_all_concurrency.sh                            # 全部默认（约 1.5–2 小时）
#   VARIANTS="baseline eagle2" bash bench_all_concurrency.sh # 只跑指定 variant
#   CONC_LIST="1 8" bash bench_all_concurrency.sh            # 只跑指定并发档
#
# 后台跑（推荐，避免 SSH 断开）：
#   setsid nohup bash bench_all_concurrency.sh > /root/autodl-tmp/logs/bench_all.out 2>&1 < /dev/null &
#   echo "PID=$!"
#   tail -f /root/autodl-tmp/logs/bench_all.out
#
# 注意：set -u 防止未定义变量；不 set -e，允许某 variant 失败时其他继续。
set -uo pipefail

VARIANTS="${VARIANTS:-baseline nextn eagle2 eagle3}"
CONC_LIST="${CONC_LIST:-1 8 32 64}"

REPO=/root/autodl-tmp/spec_decode
LOGDIR=/root/autodl-tmp/logs
OUTROOT="$REPO/outputs/glm47_concurrency"
PY=/root/autodl-tmp/conda_envs/sglang/bin/python

mkdir -p "$LOGDIR" "$OUTROOT"
COMBINED="$OUTROOT/_combined_summary.tsv"
OVERALL_LOG="$LOGDIR/bench_all_overall.log"

OVERALL_START=$(date +%s)
{
  echo "===== bench_all_concurrency.sh START $(date -u +%FT%TZ) ====="
  echo "  variants  = $VARIANTS"
  echo "  conc_list = $CONC_LIST"
  echo "  out_root  = $OUTROOT"
} | tee "$OVERALL_LOG"

# combined summary 表头（覆盖式重写）
{
  printf "variant\tconcurrency\tcompleted\tttft_p50_ms\ttpot_p50_ms\titl_p50_ms\t"
  printf "output_throughput_tok_s\trequest_throughput_req_s\tduration_s\taccept_length\n"
} > "$COMBINED"

# 探测 cgroup memory current 文件路径（v2 优先，回退 v1）
CGROUP_MEM_FILE=""
if [ -r /sys/fs/cgroup/memory.current ]; then
  CGROUP_MEM_FILE=/sys/fs/cgroup/memory.current
elif [ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
  CGROUP_MEM_FILE=/sys/fs/cgroup/memory/memory.usage_in_bytes
fi

cgroup_mem_gb() {
  if [ -z "$CGROUP_MEM_FILE" ]; then echo 0; return; fi
  local b
  b=$(cat "$CGROUP_MEM_FILE" 2>/dev/null || echo 0)
  echo $(( b / 1024 / 1024 / 1024 ))
}

# 容器 cgroup 限制 128 GB；单 sglang server 占 ~68 GB。切换时必须等到内存掉下来，
# 否则下一个 server 启动时 cgroup OOM 直接 SIGKILL（不写 dmesg）。
# 阈值默认 20 GB（保留余量给 page cache、其他进程）。
KILL_MEM_THRESHOLD_GB="${KILL_MEM_THRESHOLD_GB:-20}"
KILL_MEM_TIMEOUT_S="${KILL_MEM_TIMEOUT_S:-120}"

kill_server() {
  ps -eo pid,args | awk '/launch[_]server/ {print $1}' | xargs -r kill -9 2>/dev/null || true
  sleep 3

  # 等 GPU 显存释放
  for i in $(seq 1 30); do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$used_mib" ] || [ "$used_mib" -lt 8192 ] 2>/dev/null; then
      break
    fi
    sleep 2
  done
  echo "  [gpu] used ${used_mib:-?} MiB"

  # 等 cgroup 内存释放（关键：cgroup OOM 会 SIGKILL 新启动的 server，不写 dmesg）
  if [ -n "$CGROUP_MEM_FILE" ]; then
    local elapsed=0
    while [ "$elapsed" -lt "$KILL_MEM_TIMEOUT_S" ]; do
      cur_gb=$(cgroup_mem_gb)
      if [ "$cur_gb" -lt "$KILL_MEM_THRESHOLD_GB" ]; then
        echo "  [cgroup] mem ${cur_gb} GB < ${KILL_MEM_THRESHOLD_GB} GB after ${elapsed}s"
        return 0
      fi
      # 每 10s 主动 drop_caches，加速 page cache 回收
      if [ $((elapsed % 10)) -eq 0 ] && [ -w /proc/sys/vm/drop_caches ]; then
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
      fi
      sleep 3
      elapsed=$((elapsed + 3))
    done
    echo "  [cgroup] WARN: ${KILL_MEM_TIMEOUT_S}s 后 mem 仍为 ${cur_gb} GB（阈值 ${KILL_MEM_THRESHOLD_GB} GB），强行继续"
  fi
}

append_combined() {
  local variant="$1" c="$2" json="$3"
  if [ ! -f "$json" ]; then
    return
  fi
  "$PY" - "$json" "$variant" "$c" >> "$COMBINED" <<'PY'
import json, sys
path, variant, c = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = [l for l in open(path) if l.strip()]
d = json.loads(lines[-1])
def g(k, default=0.0):
    v = d.get(k, default)
    return v if v is not None else default
print(
    f"{variant}\t{c}\t{int(g('completed'))}"
    f"\t{g('median_ttft_ms'):.1f}"
    f"\t{g('median_tpot_ms'):.2f}"
    f"\t{g('median_itl_ms'):.2f}"
    f"\t{g('output_throughput'):.2f}"
    f"\t{g('request_throughput'):.3f}"
    f"\t{g('duration'):.1f}"
    f"\t{g('accept_length', 0):.3f}"
)
PY
}

for VARIANT in $VARIANTS; do
  V_START=$(date +%s)
  echo "" | tee -a "$OVERALL_LOG"
  echo "=================================================" | tee -a "$OVERALL_LOG"
  echo " >>> variant=$VARIANT  start=$(date -u +%FT%TZ)" | tee -a "$OVERALL_LOG"
  echo "=================================================" | tee -a "$OVERALL_LOG"

  kill_server

  LAUNCH_LOG="$LOGDIR/bench_all_launch_${VARIANT}.log"
  echo "  [launch] -> $LAUNCH_LOG" | tee -a "$OVERALL_LOG"
  if ! bash "$REPO/launch_glm47.sh" "$VARIANT" > "$LAUNCH_LOG" 2>&1; then
    echo "  [SKIP] $VARIANT 启动失败，launch log 末尾：" | tee -a "$OVERALL_LOG"
    tail -8 "$LAUNCH_LOG" | sed 's/^/    /' | tee -a "$OVERALL_LOG"
    continue
  fi
  echo "  [launch] READY" | tee -a "$OVERALL_LOG"

  BENCH_LOG="$LOGDIR/bench_all_bench_${VARIANT}.log"
  echo "  [bench]  CONC_LIST='$CONC_LIST' -> $BENCH_LOG" | tee -a "$OVERALL_LOG"
  CONC_LIST="$CONC_LIST" bash "$REPO/bench_concurrency.sh" "$VARIANT" > "$BENCH_LOG" 2>&1
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "  [WARN] bench_concurrency.sh 退出码 $RC，仍尝试收集结果" | tee -a "$OVERALL_LOG"
  fi

  for C in $CONC_LIST; do
    append_combined "$VARIANT" "$C" "$OUTROOT/$VARIANT/c${C}.json"
  done

  V_END=$(date +%s)
  echo "  [done]   variant=$VARIANT 耗时 $((V_END - V_START)) s" | tee -a "$OVERALL_LOG"
done

kill_server

OVERALL_END=$(date +%s)
ELAPSED=$((OVERALL_END - OVERALL_START))
{
  echo ""
  echo "===== ALL DONE  total $((ELAPSED/60)) min $((ELAPSED%60)) sec ====="
  echo "  combined summary: $COMBINED"
} | tee -a "$OVERALL_LOG"

echo ""
echo "==== combined summary ===="
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$COMBINED"
else
  cat "$COMBINED"
fi
