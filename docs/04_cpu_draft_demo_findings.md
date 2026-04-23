# CPU Draft Demo 初次测量发现（2026-04-23）

**一句话结论**：用当前的 `GLM-4.5-DRAFT-0.6B` 作为 CPU draft 去打 `GLM-4-32B-int4`，
accept rate 只有 17%，理论最快（overlap 模式）也只有 24 tok/s —— **打不过** sglang 自己的
STANDALONE（65 tok/s）。瓶颈不是 CPU 速度，而是 **draft 和 target 是两代模型，分布不一致**。

## 测量环境

| 项目 | 值 |
|---|---|
| Target | `glm-4-32b-0414-gptq-int4` 跑在 sglang 上（无投机，batch=1） |
| Draft | `GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf` 跑在 CPU 上 |
| Target GPU | RTX 5090 |
| Draft CPU | Intel Xeon 8470Q（Sapphire Rapids，AMX_INT8），taskset 锁到 0-31 核 |
| 输入 | 12 条真实混合场景 prompt（代码、中英问答、创作、JSON、翻译），循环 24 次 |
| `n_draft` | 5 |

脚本：`cpu_draft_demo.py`（v2，用 eval + sample 底层 API 分离 prompt eval 和 gen）。

## 原始数据

| 指标 | median |
|---|---|
| CPU draft prompt eval | 488 ms（每条新 prompt 一次性开销） |
| CPU draft gen 5 tokens | 76 ms |
| CPU draft 稳态速度 | **65 tok/s** |
| GPU target 单 token | 28.1 ms（**35.6 tok/s**，这是无投机 baseline） |
| accept length 均值 | 0.83 / 5 |
| accept rate 均值 | **16.7%** |

### accept length 分布（24 次测量）
```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 5, 5]
 ← 18 次零匹配                                   ↑ 2 次 1/5  ↑ 2 次 4/5  ↑ 2 次 5/5
```

**高度两极化**：多数完全不匹配，少数完全匹配（比如英文长文、英文对话）。这说明
draft 模型的表达分布和 target 分布在大多数场景下就是不重合的，不是微小调优能改的问题。

## 理论加速比（三档估算）

| 模式 | speedup | 绝对 tok/s | 备注 |
|---|---|---|---|
| naive（每轮都 prompt eval） | 0.09x | ~3 | 最差，等于 12x 变慢 |
| steady（prompt eval 摊薄） | **0.49x** | ~18 | 真实集成应该是这个水平 |
| overlap（CPU 和 GPU 并行） | **0.67x** | ~24 | 理论最好，Phase 1 终点能做到 |

## 对比参考

| 方案 | tok/s |
|---|---|
| 无投机 baseline | 36 |
| 本 CPU draft 方案（overlap 估算） | ~24 |
| STANDALONE（同一 draft 模型，GPU 上） | 65 |
| NGRAM（sglang 原生，模板化数据上） | 300 |

**本方案连 baseline 都打不过，更不用说 STANDALONE**。

## 为什么 CPU 速度不是问题

做个反事实分析：假设 accept rate 提到 71%（即 accept_len = 3.5），其他不变：
- E = 4.5 tokens / iter
- overlap_iter_time = max(76, 30) = 76 ms
- throughput = 4.5 / 0.076 = **59 tok/s** → 打平 STANDALONE
- 如果 accept rate 85%（accept_len = 4.25），throughput = 5.25 / 0.076 = **69 tok/s** → 略超

再看 accept rate 99%（sglang NGRAM 模板数据上的水平）：
- E = 5.95 tokens / iter
- overlap → 78 tok/s

**CPU 不是瓶颈，draft 模型的预测准确度才是**。

## 为什么 accept rate 这么低

`GLM-4.5-DRAFT-0.6B` 是**为 GLM-4.5 设计的 draft**，但 target 是 **GLM-4-32B**。
两代模型的概率分布不一致，用一代的 draft 去预测另一代，本质上就是错配。

证据：sglang 原生的 STANDALONE 用**同一对模型**，accept rate 也只有 27%（见
outputs/benchmarks/sglang_baseline/20260422_17253{2,7} 的历史记录）。

## 下一步建议

### 短期（给老板汇报用）
1. 把这份文档和 `outputs/cpu_draft_demo/cpu_draft_demo_20260423_163329/summary.json` 一起给老板
2. 确认一个关键问题：**这个项目要超过 sglang 的哪一档？**
   - 如果要超 NGRAM（300 tok/s）：当前这条路不可能，NGRAM 在模板化数据上本来就是上限
   - 如果要超 STANDALONE（65 tok/s）：需要更好的 draft 模型

### 中期（技术路径，按 ROI 排）
1. **找或训 GLM-4 专属 draft 模型**：这是最根本的解。可以考虑：
   - 从 GLM-4 体系自己蒸馏一个 0.5–1B 小模型
   - 和智谱合作直接拿官方 draft（如果他们有内部版本）
2. **EAGLE for GLM-4**：训练 EAGLE 头，历史上 EAGLE accept rate 普遍 80%+
3. **NGRAM + 小模型混合**：命中 n-gram 时用 ngram 超快，miss 时 fallback 到小模型
4. **REST（检索式投机）**：如果业务有 domain corpus，离线建索引，draft 直接查

### 如果坚持做 CPU draft
那至少：
1. 用方案 B（直接在 sglang 里改 `StandaloneWorker` 把 draft forward 迁到 CPU），
   彻底消除 llama-cpp-python 的 per-call 600ms 开销
2. 实现真正的 overlap（CPU 产下一轮 draft 时 GPU 在 verify 当前一轮）
3. 前提：先解决 draft 模型匹配度问题，否则怎么优化都打不过 STANDALONE

## 关于"真实场景"的数据

老板要求测真实情况，不要模板化数据。这次 demo 用的 prompt 覆盖了：代码续写、中英
问答、长文生成、JSON、翻译、对话。结论是这些场景下 **NGRAM 的 accept rate 会显著低于
模板化数据**（因为没有现成的 n-gram 可查）。

**下一步建议**：在同一套 realistic prompts 上跑一次 sglang NGRAM，确认 NGRAM 在真实数据
上到底还有多快（预计会从 300 tok/s 显著掉下来）。如果掉到 100 tok/s 左右，那"超过
NGRAM"的门槛就变得现实多了。
