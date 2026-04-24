#!/usr/bin/env python3
"""
在真实场景 prompt 上测 sglang STANDALONE 投机采样的 accept rate，用来和
cpu_draft_demo.py 的 CPU draft 数字做 apples-to-apples 对比。

目的：回答"把 draft 从 CPU 搬到 GPU（即 STANDALONE），accept rate 有没有
更高？"—— 如果同样低，证明瓶颈就是 draft 模型本身不匹配 target，和跑在哪
无关；如果明显更高，说明 CPU 集成里有潜在的 token 对齐问题或采样差异。

前置：sglang 必须以 --speculative-algorithm STANDALONE 启动，draft 模型为
GLM-4.5-0.6B-v3（和 CPU demo 用的是同一个模型的 GGUF 版本，权重一致）。
"""

import argparse
import json
import os
import re
import statistics
import time
from datetime import datetime

import requests


# 和 cpu_draft_demo.py 完全一致的 prompt 列表
REALISTIC_PROMPTS = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:",
    "写一个 Python 函数，输入一个字符串，返回其中每个单词出现的次数：\n\ndef word_count(s):",
    "Complete the following Rust function that checks if a number is prime:\n\nfn is_prime(n: u64) -> bool {",
    "请解释一下什么是 Transformer 架构的注意力机制，以及它为什么在自然语言处理里比 RNN 更有效？",
    "如果我在训练大语言模型时遇到 loss 突然变成 NaN，通常有哪些可能的原因？应该如何排查？",
    "量化感知训练（QAT）和后训练量化（PTQ）的主要区别是什么？实际工程中哪种更常用？",
    "The historical impact of the Industrial Revolution extends far beyond the immediate technological changes of the 18th and 19th centuries. In what ways did",
    "When designing a distributed database system that must handle millions of concurrent writes while maintaining strong consistency, the core tradeoffs involve",
    "写一个 200 字左右的短故事，讲一个迷路的机器人在森林里遇到一只会说话的乌龟。",
    "Generate a JSON object describing a user profile with fields name, email, age, preferences (as a nested object with theme and language). Example output:\n",
    "My laptop is overheating whenever I run docker containers. What should I check first?",
    "请把下面这句话翻译成流畅的英文：“投机采样的核心思想是先用一个便宜的机制猜测未来几个 token，再让大模型一次性验证。”",
]


# 匹配 sglang "Decode batch" 日志行里的 accept 指标
ACCEPT_LINE_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*?"
    r"accept len:\s*([\d.]+).*?"
    r"accept rate:\s*([\d.]+).*?"
    r"gen throughput \(token/s\):\s*([\d.]+)"
)


def call_sglang(url, prompt, max_tokens, timeout=180):
    """向 sglang 发一次贪心请求，返回 text 和完整 usage。"""
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    return {
        "text": data["choices"][0]["text"],
        "n_out": data.get("usage", {}).get("completion_tokens", max_tokens),
        "latency_s": dt,
    }


def summarize(values):
    if not values:
        return {}
    s = sorted(values)
    return {
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
        "n": len(s),
    }


def parse_log_window(path, t_start, t_end):
    """解析 sglang log 在 [t_start, t_end] 时间窗口内的 decode batch 行。"""
    hits = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if "Decode batch" not in line:
                    continue
                m = ACCEPT_LINE_RE.search(line)
                if not m:
                    continue
                try:
                    ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
                if ts < t_start - 1 or ts > t_end + 1:
                    continue
                hits.append({
                    "ts": ts,
                    "accept_len": float(m.group(2)),
                    "accept_rate": float(m.group(3)),
                    "gen_throughput": float(m.group(4)),
                })
    except FileNotFoundError:
        pass
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:6006/v1/completions")
    ap.add_argument("--server-log", required=True,
                    help="sglang server 的日志文件路径，用来抓 ground truth accept rate")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="每条 prompt 让 target 生成多少 token。太少了 decode batch 可能"
                         "还没记录到 log；64 是和 CPU demo 的 n_draft=5 大致保留可比性的"
                         "折中（实际 prompt 大小决定 decode step 数量）")
    ap.add_argument("--runs-per-prompt", type=int, default=2,
                    help="每条 prompt 跑几轮，摊平抖动")
    ap.add_argument("--warmup", type=int, default=3,
                    help="预热请求数（NGRAM/STANDALONE 都需要热 KV cache 和 tokenizer）")
    ap.add_argument("--label", default="standalone_realistic")
    ap.add_argument("--out-root", default="outputs/standalone_realistic",
                    help="输出目录根路径；NGRAM 调用时传 outputs/ngram_realistic 等")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_root, f"{args.label}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # 试探 server
    try:
        requests.get(args.url.replace("/v1/completions", "/v1/models"), timeout=10).raise_for_status()
    except Exception as e:
        raise SystemExit(f"server 探测失败: {e}")

    print(f"[{ts}] STANDALONE 真实 prompt 对比测试  label={args.label}")
    print(f"  url           : {args.url}")
    print(f"  server log    : {args.server_log}")
    print(f"  max_tokens    : {args.max_tokens}")
    print(f"  runs/prompt   : {args.runs_per_prompt}  (共 {len(REALISTIC_PROMPTS) * args.runs_per_prompt} 次测量)")
    print(f"  warmup        : {args.warmup}")

    # warmup
    for _ in range(args.warmup):
        call_sglang(args.url, REALISTIC_PROMPTS[0], args.max_tokens)

    t_bench_start = time.time()
    records = []
    for pi, prompt in enumerate(REALISTIC_PROMPTS):
        for r in range(args.runs_per_prompt):
            t0_req = time.time()
            out = call_sglang(args.url, prompt, args.max_tokens)
            t1_req = time.time()
            client_tok_per_s = out["n_out"] / out["latency_s"] if out["latency_s"] else 0.0
            rec = {
                "prompt_idx": pi,
                "run": r,
                "prompt_preview": prompt[:42].replace("\n", " "),
                "n_out": out["n_out"],
                "latency_s": out["latency_s"],
                "client_tok_per_s": client_tok_per_s,
                "t_req_start": t0_req,
                "t_req_end": t1_req,
            }
            records.append(rec)
            print(f"  [{pi:>2}/{len(REALISTIC_PROMPTS)} run{r+1}] "
                  f"{out['n_out']} tok in {out['latency_s']*1000:.0f}ms "
                  f"({client_tok_per_s:.1f} tok/s)  [{rec['prompt_preview']}]")
            # 两次请求之间稍微隔开一下，让 decode batch 日志分得清
            time.sleep(0.05)
    t_bench_end = time.time()

    # 整体窗口的 server log 解析（aggregate 视角）
    all_hits = parse_log_window(args.server_log, t_bench_start, t_bench_end)

    # 按每个请求的时间窗口分别解析（per-prompt 视角）
    per_req_hits = []
    for rec in records:
        hits = parse_log_window(args.server_log, rec["t_req_start"], rec["t_req_end"])
        per_req_hits.append({
            "prompt_idx": rec["prompt_idx"],
            "run": rec["run"],
            "n_decode_batches": len(hits),
            "accept_len_mean": statistics.mean([h["accept_len"] for h in hits]) if hits else None,
            "accept_rate_mean": statistics.mean([h["accept_rate"] for h in hits]) if hits else None,
            "gen_tps_mean": statistics.mean([h["gen_throughput"] for h in hits]) if hits else None,
        })

    # 按 prompt 聚合
    per_prompt_agg = []
    for pi in range(len(REALISTIC_PROMPTS)):
        hits_for_prompt = [h for h in per_req_hits if h["prompt_idx"] == pi]
        ac_rates = [h["accept_rate_mean"] for h in hits_for_prompt if h["accept_rate_mean"] is not None]
        ac_lens = [h["accept_len_mean"] for h in hits_for_prompt if h["accept_len_mean"] is not None]
        tps = [h["gen_tps_mean"] for h in hits_for_prompt if h["gen_tps_mean"] is not None]
        per_prompt_agg.append({
            "prompt_idx": pi,
            "prompt_preview": REALISTIC_PROMPTS[pi][:42].replace("\n", " "),
            "mean_accept_rate": statistics.mean(ac_rates) if ac_rates else None,
            "mean_accept_len": statistics.mean(ac_lens) if ac_lens else None,
            "mean_gen_tps": statistics.mean(tps) if tps else None,
        })

    summary = {
        "label": args.label,
        "timestamp": ts,
        "config": {
            "url": args.url,
            "server_log": args.server_log,
            "max_tokens": args.max_tokens,
            "runs_per_prompt": args.runs_per_prompt,
            "warmup": args.warmup,
            "num_prompts": len(REALISTIC_PROMPTS),
        },
        "aggregate": {
            "num_decode_batches_total": len(all_hits),
            "accept_len": summarize([h["accept_len"] for h in all_hits]),
            "accept_rate": summarize([h["accept_rate"] for h in all_hits]),
            "gen_throughput": summarize([h["gen_throughput"] for h in all_hits]),
        },
        "client_side": {
            "tok_per_s_per_run": summarize([r["client_tok_per_s"] for r in records]),
            "latency_ms_per_run": summarize([r["latency_s"] * 1000 for r in records]),
        },
        "per_prompt": per_prompt_agg,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "runs.json"), "w") as f:
        json.dump({"records": records, "per_req_hits": per_req_hits}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f" {args.label} — 聚合结果（sglang 日志 ground truth）")
    print("=" * 72)
    agg = summary["aggregate"]
    if agg["accept_rate"]:
        print(f"  decode batch 总数        : {agg['num_decode_batches_total']}")
        print(f"  accept length  (median)  : {agg['accept_len']['median']:.2f}  "
              f"(mean {agg['accept_len']['mean']:.2f}, min {agg['accept_len']['min']:.2f}, "
              f"max {agg['accept_len']['max']:.2f})")
        print(f"  accept rate    (median)  : {agg['accept_rate']['median']:.2%}  "
              f"(mean {agg['accept_rate']['mean']:.2%}, min {agg['accept_rate']['min']:.2%}, "
              f"max {agg['accept_rate']['max']:.2%})")
        print(f"  gen throughput (median)  : {agg['gen_throughput']['median']:.1f} tok/s  "
              f"(mean {agg['gen_throughput']['mean']:.1f})")
    else:
        print("  没抓到 accept 数据！检查 server log 路径和 sglang 是否开了投机")
    print()
    print(f"  客户端测 tok/s (median)  : {summary['client_side']['tok_per_s_per_run']['median']:.1f}")
    print()
    print(f"  ==== 按 prompt 分组的 accept rate ====")
    for p in per_prompt_agg:
        ar = f"{p['mean_accept_rate']:.1%}" if p['mean_accept_rate'] is not None else "n/a"
        al = f"{p['mean_accept_len']:.2f}" if p['mean_accept_len'] is not None else "n/a"
        print(f"    [{p['prompt_idx']:>2}] accept_rate={ar:>6}  len={al:>5}  {p['prompt_preview']}")

    print()
    print(f"  输出到: {out_dir}")


if __name__ == "__main__":
    main()
