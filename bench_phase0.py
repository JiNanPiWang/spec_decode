#!/usr/bin/env python3
"""
Phase 0 基准测试脚本：单点配置 smoke test，用于 5090 + GLM 系列模型。

目标：每次运行只回答一个问题 —— 在当前这个 sglang server 上，batch=1、
给定的 input/output 长度下，能跑多少 tok/s？

对每个你想对比的 server 变体分别跑一次：
  1) baseline    （无投机采样）
  2) sglang 原生 （NGRAM / STANDALONE / EAGLE 等）
  3) 自研方案    （比如 CPU draft）

**不要**在这里就跑完整的 (batch, seq) 矩阵。那是 Phase 2 的事，用 bench_baseline.py。

用法:
    python3 bench_phase0.py --label sglang_ngram
    python3 bench_phase0.py --label sglang_ngram --server-log /tmp/sglang_runs/ngram_xxx.log
    python3 bench_phase0.py --label my_ngram --input-chars 512 --output-tokens 128 --num-requests 10

NGRAM 的注意事项：
    NGRAM 对 trie 冷热非常敏感，accept rate 能从 0.1（冷）跳到 0.9（热）。
    默认 warmup=2 不够，建议 --warmup 5 或更多；或者用 --fixed-prompt 固定 prompt
    彻底消除 trie 冷启动抖动。
"""

import argparse
import csv
import json
import os
import random
import re
import statistics
import time
from datetime import datetime

import requests


# 匹配 sglang log 里 "Decode batch" 这一行的关键字段
ACCEPT_LINE_RE = re.compile(
    r"accept len:\s*([\d.]+).*?accept rate:\s*([\d.]+).*?gen throughput \(token/s\):\s*([\d.]+)"
)


def random_prompt(n_chars: int) -> str:
    """生成大约 n_chars 个字符的随机英文 prompt（词库小，方便控制长度）。"""
    words = [
        "speculative", "decoding", "transformer", "attention", "inference",
        "GPU", "CPU", "draft", "verify", "tokens", "latency", "throughput",
        "GLM", "sglang", "baseline", "benchmark", "pipeline", "batch",
    ]
    out = []
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(random.choice(words))
    return " ".join(out)[:n_chars]


def one_request(base_url: str, prompt: str, max_tokens: int, timeout: int = 300) -> dict:
    """向 sglang /v1/completions 发一次贪心请求，返回本次的 latency 相关指标。"""
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(base_url, json=payload, timeout=timeout)
    t1 = time.perf_counter()
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    in_tok = usage.get("prompt_tokens", 0)
    out_tok = usage.get("completion_tokens", max_tokens)
    return {
        "latency_s": t1 - t0,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "tpot_ms": (t1 - t0) * 1000 / max(out_tok, 1),
    }


def summarize_samples(values, percentiles=(50, 90, 99)):
    """对一组数值算 mean/median/stdev/min/max + 若干分位。空列表返回空 dict。"""
    if not values:
        return {}
    s = sorted(values)
    out = {
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
    }
    for p in percentiles:
        idx = min(int(len(s) * p / 100), len(s) - 1)
        out[f"p{p}"] = s[idx]
    return out


def parse_sglang_log_between(path, t_start_s, t_end_s):
    """解析 sglang server log，把 bench 时间窗口内的 "Decode batch" 行提出来。

    返回 list of dicts，包含 accept_len / accept_rate / gen_throughput。这些是
    sglang 自己汇报的 ground truth，比客户端算的更权威。"""
    hits = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                # 格式：[2026-04-23 14:36:44] Decode batch, ... accept len: X, accept rate: Y, ... gen throughput (token/s): Z
                if "Decode batch" not in line:
                    continue
                ts_match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
                if not ts_match:
                    continue
                try:
                    ts = time.mktime(time.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
                if ts < t_start_s - 1 or ts > t_end_s + 1:
                    continue
                m = ACCEPT_LINE_RE.search(line)
                if not m:
                    continue
                hits.append({
                    "ts": ts,
                    "accept_len": float(m.group(1)),
                    "accept_rate": float(m.group(2)),
                    "gen_throughput": float(m.group(3)),
                })
    except FileNotFoundError:
        pass
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--input-chars", type=int, default=512,
                    help="prompt 长度（字符，不是 token）的近似值")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--num-requests", type=int, default=10,
                    help="串行发多少次请求，取平均")
    ap.add_argument("--warmup", type=int, default=2,
                    help="预热请求数（不计入统计）。NGRAM 场景下建议 5+")
    ap.add_argument("--label", default="unlabeled",
                    help="输出目录的 label，比如 sglang_ngram / cpu_draft")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-prompt", default=None,
                    help="如果设置，每次请求都用这个固定 prompt（绕过随机生成）。"
                         "用来消除 NGRAM 的 trie 冷热抖动")
    ap.add_argument("--server-log", default=None,
                    help="sglang server log 路径；如果提供，脚本会把 log 里属于本次 "
                         "bench 时间窗口的 accept rate / gen throughput 汇总打印")
    args = ap.parse_args()

    random.seed(args.seed)

    base_url = f"http://{args.host}:{args.port}/v1/completions"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", "phase0", f"{args.label}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # 先试探 server 是否能响应
    try:
        r = requests.get(f"http://{args.host}:{args.port}/v1/models", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"探测 server 失败 {args.host}:{args.port} - {e}")

    print(f"[{ts}] Phase 0 单点 bench  label={args.label}")
    print(f"  target  : {base_url}")
    print(f"  input~{args.input_chars} chars, output={args.output_tokens} tokens, batch=1")
    print(f"  warmup={args.warmup}, runs={args.num_requests}, fixed_prompt={bool(args.fixed_prompt)}")

    def make_prompt():
        return args.fixed_prompt if args.fixed_prompt else random_prompt(args.input_chars)

    for _ in range(args.warmup):
        one_request(base_url, make_prompt(), args.output_tokens)

    bench_t_start = time.time()
    t_start_perf = time.perf_counter()
    results = []
    for i in range(args.num_requests):
        r = one_request(base_url, make_prompt(), args.output_tokens)
        results.append(r)
        tok_per_s = r["output_tokens"] / r["latency_s"] if r["latency_s"] else 0.0
        print(f"  run {i+1}/{args.num_requests}: "
              f"{r['output_tokens']} tok in {r['latency_s']*1000:.0f}ms "
              f"({tok_per_s:.1f} tok/s, tpot={r['tpot_ms']:.1f}ms)")
    t_all = time.perf_counter() - t_start_perf
    bench_t_end = time.time()

    latencies = [r["latency_s"] for r in results]
    tpots = [r["tpot_ms"] for r in results]
    tok_per_s_per_run = [r["output_tokens"] / r["latency_s"] if r["latency_s"] else 0.0
                         for r in results]
    total_out = sum(r["output_tokens"] for r in results)

    lat_stats = summarize_samples([x * 1000 for x in latencies])        # ms
    tpot_stats = summarize_samples(tpots)                               # ms
    tps_stats = summarize_samples(tok_per_s_per_run)                    # tok/s per run

    summary = {
        "label": args.label,
        "timestamp": ts,
        "config": {
            "input_chars": args.input_chars,
            "output_tokens_target": args.output_tokens,
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "batch": 1,
            "fixed_prompt_used": bool(args.fixed_prompt),
        },
        "aggregate": {
            "total_wall_s": t_all,
            "total_output_tokens": total_out,
            "avg_tok_per_s": total_out / t_all if t_all else 0.0,
        },
        "latency_ms": lat_stats,
        "tpot_ms": tpot_stats,
        "tok_per_s_per_run": tps_stats,
    }

    # 如果给了 server log，额外解析 sglang 自己汇报的 accept rate
    if args.server_log:
        hits = parse_sglang_log_between(args.server_log, bench_t_start, bench_t_end)
        if hits:
            summary["sglang_log"] = {
                "num_decode_batches": len(hits),
                "accept_len": summarize_samples([h["accept_len"] for h in hits]),
                "accept_rate": summarize_samples([h["accept_rate"] for h in hits]),
                "gen_throughput": summarize_samples([h["gen_throughput"] for h in hits]),
            }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "latency_s", "input_tokens", "output_tokens", "tpot_ms"])
        w.writeheader()
        for i, r in enumerate(results):
            w.writerow({"run": i, **r})

    print()
    print("=" * 64)
    print(f" {args.label}")
    print("=" * 64)
    print(f"  输出 token 总数      : {total_out}")
    print(f"  总 wall 时间         : {t_all:.2f}s")
    print(f"  聚合 tok/s           : {summary['aggregate']['avg_tok_per_s']:.1f}  (total_out / wall)")
    print()
    print(f"  per-run tok/s        : "
          f"median {tps_stats['median']:.1f}  mean {tps_stats['mean']:.1f}  "
          f"min {tps_stats['min']:.1f}  max {tps_stats['max']:.1f}")
    print(f"  per-run latency ms   : "
          f"median {lat_stats['median']:.0f}  mean {lat_stats['mean']:.0f}  "
          f"p90 {lat_stats['p90']:.0f}  p99 {lat_stats['p99']:.0f}")
    print(f"  per-run TPOT ms/tok  : "
          f"median {tpot_stats['median']:.2f}  mean {tpot_stats['mean']:.2f}")
    if "sglang_log" in summary:
        s = summary["sglang_log"]
        print()
        print(f"  sglang log（窗口内 {s['num_decode_batches']} 个 decode batch）:")
        print(f"    accept len       : median {s['accept_len']['median']:.2f}  "
              f"mean {s['accept_len']['mean']:.2f}  "
              f"min {s['accept_len']['min']:.2f}  max {s['accept_len']['max']:.2f}")
        print(f"    accept rate      : median {s['accept_rate']['median']:.2f}  "
              f"mean {s['accept_rate']['mean']:.2f}  "
              f"min {s['accept_rate']['min']:.2f}  max {s['accept_rate']['max']:.2f}")
        print(f"    gen throughput   : median {s['gen_throughput']['median']:.1f}  "
              f"mean {s['gen_throughput']['mean']:.1f}  "
              f"min {s['gen_throughput']['min']:.1f}  max {s['gen_throughput']['max']:.1f}")
    print()
    print(f"  输出到               : {out_dir}")


if __name__ == "__main__":
    main()
