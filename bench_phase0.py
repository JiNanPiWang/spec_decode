#!/usr/bin/env python3
"""
Phase 0 benchmark: single-config smoke test for GLM4-int4 on 5090.

Purpose:
    Only answer one question per run: on this sglang server, at batch=1,
    a fixed input/output length, how many tokens/s do we get?

    Run this script once against each server variant you want to compare:
      1) baseline (no speculative decoding)
      2) sglang native speculative (EAGLE / STANDALONE / NGRAM)
      3) your ngram variant

    Do NOT sweep (batch, seq) matrices here. Use bench_baseline.py for that
    after we've decided on a configuration worth sweeping.

Usage:
    python3 bench_phase0.py                          # defaults
    python3 bench_phase0.py --input-chars 512 --output-tokens 128 --num-requests 10
    python3 bench_phase0.py --label sglang_eagle     # writes results under this label
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime

import requests


def random_prompt(n_chars: int) -> str:
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


def percentile(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--input-chars", type=int, default=512,
                    help="Approximate prompt length in characters (not tokens).")
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--num-requests", type=int, default=10,
                    help="Total serial requests to average over.")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--label", default="unlabeled",
                    help="Tag for the output directory, e.g. 'sglang_eagle'.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    base_url = f"http://{args.host}:{args.port}/v1/completions"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", "phase0", f"{args.label}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Probe server.
    try:
        r = requests.get(f"http://{args.host}:{args.port}/v1/models", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"Server probe failed at {args.host}:{args.port} - {e}")

    print(f"[{ts}] Phase 0 single-config bench  label={args.label}")
    print(f"  target: {base_url}")
    print(f"  input~{args.input_chars} chars, output={args.output_tokens} tokens, batch=1")
    print(f"  warmup={args.warmup}, runs={args.num_requests}")

    # Warmup (not counted).
    for _ in range(args.warmup):
        one_request(base_url, random_prompt(args.input_chars), args.output_tokens)

    # Serial runs. batch=1 is enforced by not using any concurrency.
    results = []
    t_all_start = time.perf_counter()
    for i in range(args.num_requests):
        r = one_request(base_url, random_prompt(args.input_chars), args.output_tokens)
        results.append(r)
        print(f"  run {i+1}/{args.num_requests}: "
              f"{r['output_tokens']} tok in {r['latency_s']*1000:.0f}ms "
              f"({r['output_tokens']/r['latency_s']:.1f} tok/s, tpot={r['tpot_ms']:.1f}ms)")
    t_all = time.perf_counter() - t_all_start

    latencies = [r["latency_s"] for r in results]
    tpots = [r["tpot_ms"] for r in results]
    total_out = sum(r["output_tokens"] for r in results)

    summary = {
        "label": args.label,
        "timestamp": ts,
        "input_chars": args.input_chars,
        "output_tokens_target": args.output_tokens,
        "num_requests": args.num_requests,
        "batch": 1,
        "total_wall_s": t_all,
        "total_output_tokens": total_out,
        "avg_tok_per_s": total_out / t_all if t_all else 0.0,
        "avg_latency_ms": 1000 * sum(latencies) / len(latencies),
        "p50_latency_ms": 1000 * percentile(latencies, 50),
        "p99_latency_ms": 1000 * percentile(latencies, 99),
        "avg_tpot_ms": sum(tpots) / len(tpots),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "latency_s", "input_tokens", "output_tokens", "tpot_ms"])
        w.writeheader()
        for i, r in enumerate(results):
            w.writerow({"run": i, **r})

    print()
    print("=" * 60)
    print(f" {args.label}")
    print("=" * 60)
    print(f"  total output tokens : {total_out}")
    print(f"  total wall time     : {t_all:.2f}s")
    print(f"  avg tok/s (batch=1) : {summary['avg_tok_per_s']:.1f}")
    print(f"  avg TPOT            : {summary['avg_tpot_ms']:.1f} ms/token")
    print(f"  P50 / P99 latency   : {summary['p50_latency_ms']:.0f} / {summary['p99_latency_ms']:.0f} ms")
    print(f"  saved to            : {out_dir}")


if __name__ == "__main__":
    main()
