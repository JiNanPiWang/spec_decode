#!/usr/bin/env python3
"""
CPU draft + GPU target 投机采样可行性 demo（v2）。

目标：用"最小代价"回答 CPU draft 这条路能不能走通，不做真正的集成投机采样
（真正的集成要改 sglang 内部，放在 Phase 1 后续做）。

v1 的教训：直接用 llama-cpp-python 的 create_completion，每次 call 有 ~600ms 固定开销
（tokenize + KV 重置 + 调度），会把 draft 速度显著低估。
v2 改为直接用 eval + sample 的底层 API，把 "prompt eval" 和 "draft gen" 分开计时，
报告"真实集成场景下每 draft token 的增量成本"。

流程（每条 prompt）：
  1) sglang target 贪心生成 N_DRAFT 个 token，记耗时（作为金标准 + T_verify 估算）
  2) CPU draft：
     a) reset + eval(prompt_tokens) 记为 prompt_eval_ms（一次性）
     b) 循环 N_DRAFT 次 sample + eval，记为 draft_gen_ms（增量）
  3) 用 target tokenizer 对齐 draft 输出和 target 输出，算 accept_length
  4) 汇总 → 输出三档速度估算：
     - 理论上限  : 忽略 prompt eval（spec decode 稳态就是这个）
     - 带 overlap: CPU draft 与 GPU verify 并行
     - naive     : 不 overlap，prompt eval 也不摊薄（最坏情况）

前置：
  - sglang 以无投机模式启动（否则 target 的"金标准"实际是 spec 输出，有歧义）
  - llama-cpp-python 已装，GGUF draft 模型在路径里
  - 强烈建议用 taskset 绑到单个 NUMA 节点（双路 CPU 跨 socket 会让性能掉 10x+）
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime

import requests
from llama_cpp import Llama
from transformers import AutoTokenizer


# 真实场景 prompt（代码 / 中文问答 / 英文长文 / 结构化输出）
# 避免重复 token 堆；不测 ngram 友好的模板化数据
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


def summarize(values):
    """对一组数值算 mean/median/min/max/stdev。空列表返回空 dict。"""
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


def call_target_greedy(url, prompt, n_tokens, timeout=120):
    """让 sglang target 对 prompt 贪心生成 n_tokens 个 token。返回 (text, n, elapsed_s)。"""
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": n_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    dt = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    n_out = data.get("usage", {}).get("completion_tokens", n_tokens)
    return text, n_out, dt


def draft_generate_split_timing(llm, prompt_text, n_draft):
    """用 llama-cpp 的低层 API 分离 prompt eval 和 gen 的耗时。

    返回:
        out_text         : draft 的文本输出
        n_gen            : 实际生成的 token 数（应等于 n_draft）
        prompt_eval_s    : prompt 喂入 + forward 的时间（一次性）
        gen_s            : 生成 n_draft 个 token 的时间（增量）
        per_tok_ms_list  : 每个 draft token 的耗时（list），用于细粒度统计
    """
    # tokenize + reset 也算进 prompt_eval（因为集成时也得做一次）
    t0 = time.perf_counter()
    llm.reset()
    prompt_tokens = llm.tokenize(prompt_text.encode("utf-8"), add_bos=True, special=True)
    llm.eval(prompt_tokens)
    prompt_eval_s = time.perf_counter() - t0

    out_tokens = []
    per_tok = []
    t_gen_start = time.perf_counter()
    for _ in range(n_draft):
        t_t0 = time.perf_counter()
        tok = llm.sample(temp=0.0, top_k=1)
        llm.eval([tok])
        per_tok.append((time.perf_counter() - t_t0) * 1000.0)
        out_tokens.append(tok)
        # 如遇 EOS 就停
        if tok == llm.token_eos():
            break
    gen_s = time.perf_counter() - t_gen_start

    try:
        out_text = llm.detokenize(out_tokens).decode("utf-8", errors="replace")
    except Exception:
        out_text = ""

    return out_text, len(out_tokens), prompt_eval_s, gen_s, per_tok


def token_level_match(target_tokenizer, draft_text, target_text):
    """用 target tokenizer 把两段文本切成 token id 序列，返回前缀匹配长度。

    为什么用 target tokenizer：最终 target verify 按 target tokenizer 算，
    所以我们应该以 target 的视角评估 accept length。"""
    draft_ids = target_tokenizer.encode(draft_text, add_special_tokens=False)
    target_ids = target_tokenizer.encode(target_text, add_special_tokens=False)
    n = 0
    for a, b in zip(draft_ids, target_ids):
        if a == b:
            n += 1
        else:
            break
    return n, len(draft_ids), len(target_ids)


def estimate_speedups(accept_len_mean, n_draft,
                      t_prompt_eval_s, t_draft_gen_total_s, t_target_verify_s):
    """三档加速比估算。

    一次迭代产出 E = (accept_len_mean + 1) 个 token，对照无投机情况下花 E × T_verify。

    - naive    : draft_total = prompt_eval + gen；verify 串行；最差场景
    - steady   : 假设 prompt eval 已经在上一轮摊薄，draft_total ≈ gen only；串行
    - overlap  : 假设 CPU draft 和 GPU verify 重叠；一次迭代时间 = max(draft, verify)
    """
    E = accept_len_mean + 1
    baseline_time = E * t_target_verify_s

    def spd(iter_time):
        return baseline_time / iter_time if iter_time > 0 else 0.0

    naive_iter_time = t_prompt_eval_s + t_draft_gen_total_s + t_target_verify_s
    steady_iter_time = t_draft_gen_total_s + t_target_verify_s
    overlap_iter_time = max(t_draft_gen_total_s, t_target_verify_s)

    return {
        "naive_speedup": spd(naive_iter_time),
        "steady_speedup": spd(steady_iter_time),
        "overlap_speedup": spd(overlap_iter_time),
        "baseline_per_iter_s": baseline_time,
        "naive_iter_s": naive_iter_time,
        "steady_iter_s": steady_iter_time,
        "overlap_iter_s": overlap_iter_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-gguf", required=True)
    ap.add_argument("--target-model-path", required=True)
    ap.add_argument("--target-url", default="http://127.0.0.1:6006/v1/completions")
    ap.add_argument("--n-draft", type=int, default=5)
    ap.add_argument("--n-iters", type=int, default=24)
    ap.add_argument("--cpu-threads", type=int, default=32)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=3,
                    help="draft 和 target 各 warmup N 次（不计数）。"
                         "CPU draft 首次调用有 ~20s 初始化开销，至少 2 次。")
    ap.add_argument("--label", default="cpu_draft_demo")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", "cpu_draft_demo", f"{args.label}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{ts}] CPU draft demo v2  label={args.label}")
    print(f"  draft gguf : {args.draft_gguf}")
    print(f"  target url : {args.target_url}")
    print(f"  n_draft    : {args.n_draft}  n_iters : {args.n_iters}")
    print(f"  CPU threads: {args.cpu_threads}  n_ctx : {args.n_ctx}")

    # 加载 CPU draft
    print("\n[step 1] 加载 CPU draft ...")
    t0 = time.perf_counter()
    draft = Llama(
        model_path=args.draft_gguf,
        n_ctx=args.n_ctx,
        n_threads=args.cpu_threads,
        n_gpu_layers=0,
        logits_all=False,
        verbose=False,
    )
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print("\n[step 2] 加载 target tokenizer ...")
    target_tok = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    print(f"  vocab={target_tok.vocab_size}")

    print(f"\n[step 3] warmup × {args.warmup} ...")
    for _ in range(args.warmup):
        draft_generate_split_timing(draft, REALISTIC_PROMPTS[0], args.n_draft)
        call_target_greedy(args.target_url, REALISTIC_PROMPTS[0], args.n_draft)

    print(f"\n[step 4] 测量 × {args.n_iters} ...")
    records = []
    prompts_cycled = (REALISTIC_PROMPTS * ((args.n_iters + len(REALISTIC_PROMPTS) - 1)
                                           // len(REALISTIC_PROMPTS)))[:args.n_iters]
    for i, prompt in enumerate(prompts_cycled):
        # target 贪心生成
        tgt_text, tgt_n, tgt_dt = call_target_greedy(args.target_url, prompt, args.n_draft)
        # CPU draft，分开测 prompt eval 和 gen
        drf_text, drf_n, pe_s, gen_s, per_tok_ms = draft_generate_split_timing(
            draft, prompt, args.n_draft
        )
        match_len, _, _ = token_level_match(target_tok, drf_text, tgt_text)

        rec = {
            "iter": i,
            "prompt_preview": prompt[:42].replace("\n", " "),
            "draft_tokens_out": drf_n,
            "target_tokens_out": tgt_n,
            "match_len": match_len,
            "draft_prompt_eval_ms": pe_s * 1000,
            "draft_gen_total_ms": gen_s * 1000,
            "draft_gen_per_tok_ms": (gen_s * 1000 / drf_n) if drf_n else 0.0,
            "draft_gen_tok_per_s": (drf_n / gen_s) if gen_s else 0.0,
            "target_total_ms": tgt_dt * 1000,
            "target_per_tok_ms": (tgt_dt * 1000 / tgt_n) if tgt_n else 0.0,
            "target_tok_per_s": (tgt_n / tgt_dt) if tgt_dt else 0.0,
        }
        records.append(rec)
        print(f"  iter {i+1:>2}/{args.n_iters}: "
              f"match {match_len}/{args.n_draft}  "
              f"draft[pe={pe_s*1000:.0f}ms gen={gen_s*1000:.0f}ms={rec['draft_gen_tok_per_s']:.0f}tps]  "
              f"target[{tgt_dt*1000:.0f}ms={rec['target_tok_per_s']:.0f}tps]  "
              f"[{rec['prompt_preview']}]")

    # 汇总
    match_lens = [r["match_len"] for r in records]
    gen_tok_per_s = [r["draft_gen_tok_per_s"] for r in records]
    pe_ms = [r["draft_prompt_eval_ms"] for r in records]
    gen_total_ms = [r["draft_gen_total_ms"] for r in records]
    target_tok_per_s = [r["target_tok_per_s"] for r in records]
    target_per_tok_ms = [r["target_per_tok_ms"] for r in records]

    accept_len_mean = statistics.mean(match_lens)
    accept_rate_mean = accept_len_mean / args.n_draft

    # 耗时估算用 median，抗 outlier
    t_pe_s = statistics.median(pe_ms) / 1000.0
    t_drf_gen_s = statistics.median(gen_total_ms) / 1000.0
    t_verify_s = statistics.median(target_per_tok_ms) / 1000.0

    sp = estimate_speedups(accept_len_mean, args.n_draft, t_pe_s, t_drf_gen_s, t_verify_s)

    summary = {
        "label": args.label,
        "timestamp": ts,
        "config": {
            "n_draft": args.n_draft,
            "n_iters": args.n_iters,
            "cpu_threads": args.cpu_threads,
            "n_ctx": args.n_ctx,
            "draft_gguf": args.draft_gguf,
            "target_url": args.target_url,
            "target_model_path": args.target_model_path,
        },
        "accept": {
            "mean_match_len": accept_len_mean,
            "mean_accept_rate": accept_rate_mean,
            "match_len_distribution": summarize(match_lens),
            "per_iter_match_len": match_lens,
        },
        "draft_cpu": {
            "prompt_eval_ms": summarize(pe_ms),
            "gen_total_ms_per_K": summarize(gen_total_ms),
            "gen_tok_per_s_steady": summarize(gen_tok_per_s),
        },
        "target_gpu": {
            "tok_per_s": summarize(target_tok_per_s),
            "per_tok_ms": summarize(target_per_tok_ms),
        },
        "speedup_estimates": sp,
        "notes": (
            "speedup 的三档解释："
            "naive=每轮都 prompt eval，最差；"
            "steady=prompt eval 已经在上一轮摊薄（接近真实集成），"
            "overlap=CPU draft 与 GPU verify 并行（需要架构支持，是 phase 1 步骤 4 的目标）。"
        ),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "runs.json"), "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f" {args.label}")
    print("=" * 72)
    print(f"  accept_length (mean)      : {accept_len_mean:.2f} / {args.n_draft}  "
          f"(accept_rate = {accept_rate_mean:.1%})")
    print(f"  accept_length 分布        : {sorted(match_lens)}")
    print()
    print(f"  CPU draft prompt eval     : median {statistics.median(pe_ms):.0f} ms")
    print(f"  CPU draft gen {args.n_draft} tokens    : median {statistics.median(gen_total_ms):.0f} ms "
          f"({statistics.median(gen_tok_per_s):.0f} tok/s steady)")
    print(f"  GPU target 单 token       : median {statistics.median(target_per_tok_ms):.1f} ms "
          f"({statistics.median(target_tok_per_s):.1f} tok/s)")
    print()
    print(f"  ==== 加速比估算（vs 无投机 baseline，在当前 target 上） ====")
    print(f"  naive    (每轮都 prompt eval) : {sp['naive_speedup']:.2f}x")
    print(f"  steady   (prompt eval 摊薄)   : {sp['steady_speedup']:.2f}x  ← 真实集成场景")
    print(f"  overlap  (CPU/GPU 并行执行)   : {sp['overlap_speedup']:.2f}x  ← phase 1 最终形态的上限")
    print()
    print(f"  对比参考（都是在本机同一 target 上的数字）:")
    print(f"    无投机 baseline     : ~36 tok/s")
    print(f"    STANDALONE (sglang) : ~65 tok/s  (相当于 1.8x)")
    print(f"    NGRAM (sglang 模板数据): ~300 tok/s (相当于 8x)")
    print(f"  本 CPU draft 方案")
    print(f"    steady 估算         : ~{36 * sp['steady_speedup']:.0f} tok/s")
    print(f"    overlap 估算        : ~{36 * sp['overlap_speedup']:.0f} tok/s")
    print()
    print(f"  保存到: {out_dir}")


if __name__ == "__main__":
    main()
