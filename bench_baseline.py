#!/usr/bin/env python3
"""
SGLang Baseline Benchmark — 无需 HuggingFace，纯本地运行
用法: python3 bench_baseline.py
"""

import requests
import time
import json
import csv
import random
import string
import threading
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 配置 ==========
HOST = "127.0.0.1"
PORT = 30000
BASE_URL = f"http://{HOST}:{PORT}/v1/completions"

# 测试矩阵: (并发数, 输入字符数≈token数, 输出token数)
TEST_CASES = [
    # (concurrency, input_chars, output_tokens)
    (1,  256,  64),
    (1,  512,  128),
    (4,  256,  64),
    (4,  512,  128),
    (8,  256,  64),
    (8,  512,  128),
    (16, 256,  64),
    (16, 512,  128),
]

NUM_REQUESTS = 5     # 每组总请求数
WARMUP_REQS  = 2    # 预热请求数（不计入结果）

# ========== 文件路径（带时间戳）==========
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = f"./bench_results_{TIMESTAMP}"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILE     = f"{OUT_DIR}/{TIMESTAMP}_results.csv"
DETAIL_LOG   = f"{OUT_DIR}/{TIMESTAMP}_detail.log"
SUMMARY_LOG  = f"{OUT_DIR}/{TIMESTAMP}_summary.log"

# ========== 工具函数 ==========

def random_prompt(n_chars: int) -> str:
    """生成随机中英文混合 prompt"""
    words = [
        "请介绍一下", "人工智能", "深度学习", "自然语言处理", "大模型", "推理加速",
        "投机采样", "transformer", "attention机制", "显卡", "并行计算",
        "请详细解释", "优化方法", "性能测试", "基准测试", "吞吐量", "延迟",
    ]
    result = []
    while sum(len(w) for w in result) < n_chars:
        result.append(random.choice(words))
    return "".join(result)[:n_chars]


def single_request(prompt: str, max_tokens: int) -> dict:
    """发送单个请求，返回延迟等指标"""
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(BASE_URL, json=payload, timeout=300)
        t1 = time.perf_counter()
        resp.raise_for_status()
        data = resp.json()
        output_tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
        input_tokens  = data.get("usage", {}).get("prompt_tokens", 0)
        return {
            "success": True,
            "latency_ms": (t1 - t0) * 1000,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tpot_ms": (t1 - t0) * 1000 / max(output_tokens, 1),
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {
            "success": False,
            "latency_ms": (t1 - t0) * 1000,
            "error": str(e),
            "input_tokens": 0,
            "output_tokens": 0,
            "tpot_ms": 0,
        }


def run_concurrent(prompts: list, max_tokens: int, concurrency: int) -> tuple:
    """并发跑一批请求，返回 (结果列表, 总耗时)"""
    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(single_request, p, max_tokens) for p in prompts]
        for f in as_completed(futures):
            results.append(f.result())
    t_end = time.perf_counter()
    return results, t_end - t_start


def percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def log(msg, also_print=True):
    with open(DETAIL_LOG, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    if also_print:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ========== 主流程 ==========

def main():
    log(f"{'='*55}")
    log(f"SGLang Baseline Benchmark  {TIMESTAMP}")
    log(f"Server : {BASE_URL}")
    log(f"Output : {OUT_DIR}")
    log(f"{'='*55}")

    # 检查 server
    try:
        r = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=10)
        log(f"Server OK: {r.status_code}")
    except Exception as e:
        log(f"ERROR: Server 未响应 — {e}")
        return

    # 写 CSV 表头
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "concurrency", "input_chars", "output_tokens",
            "num_requests", "success_count", "fail_count",
            "total_time_s", "throughput_req_s", "throughput_tok_s",
            "avg_latency_ms", "p50_latency_ms", "p90_latency_ms", "p99_latency_ms",
            "avg_tpot_ms", "total_input_tokens", "total_output_tokens",
        ])

    # 预热
    log(f"\n--- 预热 {WARMUP_REQS} 个请求 ---")
    warmup_prompts = [random_prompt(256) for _ in range(WARMUP_REQS)]
    run_concurrent(warmup_prompts, 64, concurrency=4)
    log("预热完成\n")

    summary_rows = []

    for (concurrency, input_chars, output_tokens) in TEST_CASES:
        label = f"conc={concurrency:>2} in={input_chars:>4} out={output_tokens:>3}"
        log(f"--- {label} ---")

        prompts = [random_prompt(input_chars) for _ in range(NUM_REQUESTS)]
        results, total_time = run_concurrent(prompts, output_tokens, concurrency)

        ok      = [r for r in results if r["success"]]
        fail    = [r for r in results if not r["success"]]
        lats    = [r["latency_ms"] for r in ok]
        tpots   = [r["tpot_ms"]    for r in ok]
        tot_in  = sum(r["input_tokens"]  for r in ok)
        tot_out = sum(r["output_tokens"] for r in ok)

        throughput_req = len(ok) / total_time if total_time > 0 else 0
        throughput_tok = tot_out / total_time if total_time > 0 else 0
        avg_lat  = sum(lats) / len(lats) if lats else 0
        p50      = percentile(lats, 50)
        p90      = percentile(lats, 90)
        p99      = percentile(lats, 99)
        avg_tpot = sum(tpots) / len(tpots) if tpots else 0

        log(f"  成功/失败: {len(ok)}/{len(fail)}")
        log(f"  总耗时:    {total_time:.2f}s")
        log(f"  吞吐量:    {throughput_req:.2f} req/s  |  {throughput_tok:.1f} tok/s")
        log(f"  延迟:      avg={avg_lat:.0f}ms  p50={p50:.0f}ms  p90={p90:.0f}ms  p99={p99:.0f}ms")
        log(f"  TPOT:      avg={avg_tpot:.1f}ms/token")
        if fail:
            log(f"  失败原因:  {fail[0].get('error','unknown')}")

        row = [
            TIMESTAMP, concurrency, input_chars, output_tokens,
            NUM_REQUESTS, len(ok), len(fail),
            round(total_time, 3), round(throughput_req, 3), round(throughput_tok, 1),
            round(avg_lat, 1), round(p50, 1), round(p90, 1), round(p99, 1),
            round(avg_tpot, 2), tot_in, tot_out,
        ]
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow(row)
        summary_rows.append(row)

    # 打印汇总表
    summary = []
    summary.append(f"\n{'='*90}")
    summary.append(f"  Benchmark 汇总  —  {TIMESTAMP}")
    summary.append(f"{'='*90}")
    summary.append(f"{'Conc':>6} {'In':>6} {'Out':>5} {'OK':>5} {'Time(s)':>8} {'Req/s':>7} {'Tok/s':>8} {'Avg(ms)':>9} {'P90(ms)':>9} {'P99(ms)':>9} {'TPOT(ms)':>10}")
    summary.append(f"{'-'*90}")
    for r in summary_rows:
        # r索引: 0=ts,1=conc,2=in,3=out,4=num,5=ok,6=fail,
        #        7=total_time,8=req/s,9=tok/s,
        #        10=avg_lat,11=p50,12=p90,13=p99,14=tpot,15=tot_in,16=tot_out
        summary.append(
            f"{r[1]:>6} {r[2]:>6} {r[3]:>5} {r[5]:>5} "
            f"{r[7]:>8} {r[8]:>7} {r[9]:>8} {r[10]:>9} {r[12]:>9} {r[13]:>9} {r[14]:>10}"
        )
    summary.append(f"{'='*90}")
    summary.append(f"CSV  : {CSV_FILE}")
    summary.append(f"Log  : {DETAIL_LOG}")

    summary_text = "\n".join(summary)
    print(summary_text)
    with open(SUMMARY_LOG, "w") as f:
        f.write(summary_text + "\n")

    log(f"\n所有结果已保存到 {OUT_DIR}/")


if __name__ == "__main__":
    main()