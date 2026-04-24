#!/usr/bin/env python3
"""
CPU draft + GPU target 投机采样 —— 多轮（multi-round）accept rate 测量版。

动机：v2 demo 发现 accept rate 只有 17%，但 sglang STANDALONE 在完全相同的
模型对 + 相同 prompt 下测出 40%。差距来自测量方式：

- v2 demo 是"单轮"：每条 prompt 只测"target 从 prompt 第一个 token / draft 从
  prompt 第一个 token"的分歧，一旦第一 token 不一致，后面全错。
- STANDALONE 实际是"多轮"：draft 错了之后 target 给出正确 token，下一轮 draft
  从 target 的新 context 再起步，通常又能对上几个。40% 是对多轮求平均。

本脚本：对每条 prompt 跑 R 轮 spec decode 模拟。每一轮：
  1. 用当前 context（prompt + 之前累积的 target 输出）作为输入
  2. target 贪心生成 K 个 token -> gold
  3. draft 贪心生成 K 个 token -> guess
  4. 统计 guess 和 gold 的前缀匹配长度
  5. context 前进 K 个 gold token（模拟 spec decode 的状态机）

这和 sglang STANDALONE 的内部行为高度对齐，accept rate 应该可直接对比。
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


def target_greedy(url, prompt, n_tokens, timeout=120):
    """sglang target 贪心生成 n_tokens 个 token。"""
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
    return data["choices"][0]["text"], data.get("usage", {}).get("completion_tokens", n_tokens), dt


def draft_greedy(llm, prompt_text, n_tokens):
    """CPU draft 贪心生成 n_tokens 个 token，返回 (text, gen_time_s)。

    用底层 eval+sample 以便把 prompt eval 和 gen 分开算。"""
    llm.reset()
    prompt_tokens = llm.tokenize(prompt_text.encode("utf-8"), add_bos=True, special=True)
    llm.eval(prompt_tokens)
    t0 = time.perf_counter()
    out_tokens = []
    for _ in range(n_tokens):
        t = llm.sample(temp=0.0, top_k=1)
        out_tokens.append(t)
        llm.eval([t])
        if t == llm.token_eos():
            break
    gen_s = time.perf_counter() - t0
    text = llm.detokenize(out_tokens).decode("utf-8", errors="replace")
    return text, gen_s


def match_prefix_token_ids(target_tokenizer, draft_text, target_text):
    """用 target tokenizer 把两段文本切成 token id，返回共同前缀长度。"""
    a = target_tokenizer.encode(draft_text, add_special_tokens=False)
    b = target_tokenizer.encode(target_text, add_special_tokens=False)
    n = 0
    for x, y in zip(a, b):
        if x == y:
            n += 1
        else:
            break
    return n, len(a), len(b)


def run_multi_round(url, llm, target_tok, prompt, n_draft, n_rounds):
    """对一条 prompt 跑 n_rounds 轮 spec decode 模拟。返回每轮的 match_len 列表
    + 累积的 gold target 输出 text（方便调试）。"""
    accumulated_target_text = ""
    rounds = []
    for r in range(n_rounds):
        # 当前 context = 原 prompt + 之前累积的 target 输出
        current_prompt = prompt + accumulated_target_text
        tgt_text, tgt_n, tgt_dt = target_greedy(url, current_prompt, n_draft)
        if tgt_n <= 0 or tgt_text == "":
            # target 生成完了（EOS），结束
            break
        drf_text, drf_dt = draft_greedy(llm, current_prompt, n_draft)
        m, _, _ = match_prefix_token_ids(target_tok, drf_text, tgt_text)
        rounds.append({
            "round": r,
            "match_len": m,
            "draft_text": drf_text,
            "target_text": tgt_text,
            "draft_gen_s": drf_dt,
            "target_latency_s": tgt_dt,
        })
        # 前进：把 target 这一轮的输出加到 context 里
        accumulated_target_text += tgt_text
    return rounds, accumulated_target_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-gguf", required=True)
    ap.add_argument("--target-model-path", required=True)
    ap.add_argument("--target-url", default="http://127.0.0.1:6006/v1/completions")
    ap.add_argument("--n-draft", type=int, default=5,
                    help="每轮 draft/target 生成多少 token；和 sglang 的 "
                         "--speculative-num-steps 可类比")
    ap.add_argument("--n-rounds-per-prompt", type=int, default=10,
                    help="对每条 prompt 跑几轮；10 轮会生成 ~50 个 token")
    ap.add_argument("--cpu-threads", type=int, default=32)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--label", default="cpu_draft_multiround")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", "cpu_draft_demo", f"{args.label}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{ts}] CPU draft multi-round  label={args.label}")
    print(f"  draft gguf : {args.draft_gguf}")
    print(f"  target url : {args.target_url}")
    print(f"  n_draft={args.n_draft}  rounds/prompt={args.n_rounds_per_prompt}  "
          f"prompts={len(REALISTIC_PROMPTS)}  "
          f"expected total rounds={len(REALISTIC_PROMPTS) * args.n_rounds_per_prompt}")

    print("\n[加载 CPU draft]")
    t0 = time.perf_counter()
    draft = Llama(model_path=args.draft_gguf, n_ctx=args.n_ctx,
                  n_threads=args.cpu_threads, n_gpu_layers=0, verbose=False)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    print("[加载 target tokenizer]")
    target_tok = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    print(f"  vocab={target_tok.vocab_size}")

    print(f"[warmup × {args.warmup}]")
    for _ in range(args.warmup):
        draft_greedy(draft, REALISTIC_PROMPTS[0], args.n_draft)
        target_greedy(args.target_url, REALISTIC_PROMPTS[0], args.n_draft)

    all_records = []
    per_prompt_stats = []
    for pi, prompt in enumerate(REALISTIC_PROMPTS):
        print(f"\n[prompt {pi+1}/{len(REALISTIC_PROMPTS)}] {prompt[:50].replace(chr(10), ' ')!r}")
        rounds, _ = run_multi_round(args.target_url, draft, target_tok, prompt,
                                    args.n_draft, args.n_rounds_per_prompt)
        for r in rounds:
            r["prompt_idx"] = pi
            all_records.append(r)
        match_lens = [r["match_len"] for r in rounds]
        if match_lens:
            p_mean_match = statistics.mean(match_lens)
            p_rate = p_mean_match / args.n_draft
            draft_tps = [args.n_draft / r["draft_gen_s"] if r["draft_gen_s"] else 0.0 for r in rounds]
            per_prompt_stats.append({
                "prompt_idx": pi,
                "prompt_preview": prompt[:42].replace("\n", " "),
                "n_rounds": len(rounds),
                "mean_match_len": p_mean_match,
                "mean_accept_rate": p_rate,
                "match_len_distribution": sorted(match_lens),
                "mean_draft_gen_tps": statistics.mean(draft_tps),
            })
            # 轮内摘要打印
            for r in rounds:
                print(f"    r{r['round']}: match {r['match_len']}/{args.n_draft}  "
                      f"draft_gen={r['draft_gen_s']*1000:.0f}ms")

    # 全局汇总
    all_match = [r["match_len"] for r in all_records]
    all_draft_s = [r["draft_gen_s"] for r in all_records]
    all_target_s = [r["target_latency_s"] for r in all_records]

    overall_mean_match = statistics.mean(all_match) if all_match else 0
    overall_rate = overall_mean_match / args.n_draft if all_match else 0

    summary = {
        "label": args.label,
        "timestamp": ts,
        "config": {
            "n_draft": args.n_draft,
            "n_rounds_per_prompt": args.n_rounds_per_prompt,
            "num_prompts": len(REALISTIC_PROMPTS),
            "total_rounds": len(all_records),
            "cpu_threads": args.cpu_threads,
            "draft_gguf": args.draft_gguf,
            "target_url": args.target_url,
        },
        "overall": {
            "mean_match_len": overall_mean_match,
            "mean_accept_rate": overall_rate,
            "match_len_distribution": summarize(all_match),
            "draft_gen_s_per_round": summarize(all_draft_s),
            "target_latency_s_per_round": summarize(all_target_s),
        },
        "per_prompt": per_prompt_stats,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "rounds.json"), "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f" {args.label} 汇总")
    print("=" * 72)
    print(f"  总轮数                  : {len(all_records)}")
    print(f"  overall accept rate     : {overall_rate:.2%}  (mean match {overall_mean_match:.2f} / {args.n_draft})")
    md = summary["overall"]["match_len_distribution"]
    if md:
        print(f"  match_len 分布          : mean {md['mean']:.2f}  median {md['median']:.2f}  "
              f"min {md['min']}  max {md['max']}")
    print()
    print(f"  draft gen (5 tok) median: {statistics.median([s*1000 for s in all_draft_s]):.0f} ms "
          f"({args.n_draft / statistics.median(all_draft_s):.0f} tok/s)")
    print(f"  target latency median   : {statistics.median([s*1000 for s in all_target_s]):.0f} ms")
    print()
    print(f"  ==== 按 prompt 分组 ====")
    for p in per_prompt_stats:
        print(f"    [{p['prompt_idx']:>2}] accept_rate={p['mean_accept_rate']:.1%}  "
              f"len={p['mean_match_len']:.2f}  "
              f"分布={p['match_len_distribution']}  {p['prompt_preview']}")

    # 和 STANDALONE 对比
    print()
    print(f"  ==== 对比参考 ====")
    print(f"  sglang STANDALONE (同一对模型，同一套 prompt): accept_rate ≈ 40%")
    print(f"  本 CPU draft multi-round 版本             : accept_rate ≈ {overall_rate:.1%}")
    if overall_rate > 0.3:
        print(f"  差距缩小到可接受范围，'单轮 vs 多轮'假说成立")
    else:
        print(f"  仍显著低于 STANDALONE，可能还有其他实现差异")

    print(f"\n  保存到: {out_dir}")


if __name__ == "__main__":
    main()
