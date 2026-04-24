# Realistic Prompt Benchmark Suite

这份文档是**以后所有 spec decode 方案对比的权威基线**。新方案跑完请按同一个
prompt 集和同一份脚本测，数字才可比。

## 配置

| 项 | 值 |
|---|---|
| Target 模型 | `/root/autodl-tmp/models/glm-4-32b-0414-gptq-int4` |
| Draft 模型（STANDALONE/CPU draft） | `/root/autodl-tmp/models/GLM-4.5-0.6B-v3`（GGUF F16 版本用于 CPU） |
| 硬件 | RTX 5090 + Xeon 8470Q（Sapphire Rapids, AMX） |
| Batch | 1（串行请求，单 request 场景，和老板前期要评估的点一致） |
| max_tokens | 64 per request |
| Prompts | 12 个 realistic 长 prompt，覆盖：代码补全（Py/Rust）、长篇中英文问答、JSON 生成、debug、翻译、写作 |
| Runs per prompt | 2（第 2 次会 prefix cache hit，client tok/s 不可信；用 server log 里的 decode batch 数字） |
| Warmup | 3 |

**Prompts 锁定在 `bench_standalone_realistic.py` 的 `REALISTIC_PROMPTS` 变量里**，
以后改 prompt 集就要重跑所有基线并在这里记录版本。

## 如何跑一个新方案

1. 起 sglang server（配好目标投机算法 / 无投机）
2. 记下 server 日志路径（里面有 `Decode batch ... accept len ... accept rate ...`）
3. 跑 bench：
   ```bash
   python3 bench_standalone_realistic.py \
     --url http://127.0.0.1:6006/v1/completions \
     --server-log /tmp/sglang_runs/<your_log> \
     --label <scheme_name> \
     --out-root outputs/<scheme_name>_realistic \
     --runs-per-prompt 2 --warmup 3 --max-tokens 64
   ```
4. 数字在 server 侧（从日志 parse 的 ground-truth）和客户端侧两份
5. 新增一列到下面的"基线汇总"表，并把输出目录补进最后一节

## 基线汇总（截至 2026-04-24）

| 方案 | accept rate (median) | accept_len | gen throughput (median, tok/s) | 备注 |
|---|---|---|---|---|
| baseline 无投机 | — | — | ~36 | 纯 target，估算 |
| STANDALONE (GPU draft 0.6B) | 40% | 2.77 / 5 | 38.7 | 几乎无加速 |
| **NGRAM (sglang)** | 30.5% | 2.44 / 8 | **89.4** | **当前最强 baseline，超越目标** |
| CPU draft multi-round (模拟) | 47.3% (mean) | 2.37 / 5 | — | accept 数字比较用，不是端到端 |
| CPU draft overlap（估算） | — | — | ~60 | Phase 1 集成前的理论上限 |

> **accept rate 口径注意**：分母是 `num_draft_tokens`，不同方案这个不一样，所以单
> 看 rate 没意义；**要看 accept_len 绝对值**，它才反映"每轮 iter 能多产出多少 token"。
> NGRAM 的 rate 低不代表差，它的 draft token 数更多，accept_len 反而接近其它方案。

## 为什么不信 client-side tok/s

`bench_standalone_realistic.py` 里的 `client_side.tok_per_s_per_run` 在 NGRAM 下
median 会 >200 tok/s，**这是 sglang prefix cache 命中导致的假象**（第二次同 prompt
完全不解码）。只信 `aggregate.gen_throughput`（从 server log 里 parse 的 decode
batch 真实速率）。

## 为什么旧的 "NGRAM 228 tok/s" 不算数

之前 `docs/00` / `docs/01` 里的 NGRAM 数字是 sglang 自带 `bench_serving.py` 的
`--dataset-name random`/`sharegpt` 跑出来的，prompt 高度重复且短，**n-gram trie
命中率接近 1**。在本套 realistic prompt 下 NGRAM 会掉到 89 tok/s，这才是对"自研
方案要超越 sglang"有意义的参考点。

## 历史数据目录

按时间戳顺序列出，每次更新 benchmark 就把新目录追加到这里：

- `outputs/standalone_realistic/standalone_realistic_20260423_165953/` — STANDALONE 基线 (2026-04-23)
- `outputs/ngram_realistic/ngram_realistic_20260424_112724/` — NGRAM 基线 (2026-04-24)
- `outputs/cpu_draft_demo/cpu_draft_F16_multiround_20260423_171928/` — CPU draft multi-round accept 质量 (2026-04-23)
