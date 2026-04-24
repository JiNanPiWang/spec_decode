# CPU Draft Demo 测量发现（2026-04-23，v2）

> **版本说明**：本文档在 2026-04-23 同一天经历两次重要更新：
> - 初版（v1）结论"CPU draft 和 target 不匹配，方向不可行"是**错误**的
> - 多轮测量（multi-round）重做后发现 CPU draft accept rate 约 47%，**和 sglang
>   STANDALONE 同量级**；初版 17% 的低值是"单轮测量"的系统性偏差
> - 本文档反映的是修正后的结论

## 一句话结论

**CPU draft + GPU target 技术上可行**。accept rate 47%，和 sglang STANDALONE 在相同模型对、
相同数据上测出的 46% 几乎一致。理论最优（KV 缓存保留 + CPU/GPU 并行）上限约 **60 tok/s**，
**高于** STANDALONE 在真实数据上的 38.7 tok/s。

## 经过

### 第一轮：单轮测量（cpu_draft_demo.py v2）
- accept rate 16.7%，看起来 draft/target 根本不匹配
- 于是推测是"`GLM-4.5-DRAFT-0.6B` 不适合给 `GLM-4` 当 draft"
- 对比 F16 draft：accept rate 反而降到 6.7%，说明问题不在量化

### 第二轮：对照 sglang STANDALONE（bench_standalone_realistic.py）
- 同一对模型、同一套 prompt、同一个 target
- **STANDALONE 测出的 accept rate 是 40% (median) / 46% (mean)**
- 2.4× 的差距从哪来？

### 第三轮：side-by-side 文本对比
对 prompt `def binary_search(arr, target):`：
```
TARGET : ' \n    low = 0\n    high ='
DRAFT  : ' \n    left = 0\n    right ='
```
两个模型都写出正确的 binary_search，只是一个用 `low/high`，一个用 `left/right` —— 第一个
token 就产生分歧，之后所有 token 在单轮里都算"不匹配"。

### 第四轮：多轮测量（cpu_draft_demo_multiround.py）
模拟真正的 spec decode 行为：draft 错了之后，把 target 的正确 token 加入 context，下一轮
重起。这等价于 sglang STANDALONE 的内部逻辑。
- **120 轮测量，overall accept rate 47.3%**
- 和 STANDALONE 的 46.3% 几乎一致

## 最终数字（5090 + Xeon 8470Q，12 条真实混合 prompt × 10 轮/prompt）

| 指标 | CPU draft multi-round | sglang STANDALONE |
|---|---|---|
| 模型对 | GLM-4-32B-int4 target + GLM-4.5-0.6B-F16 draft | 同 |
| accept rate | **47.3%** | 46.3% (mean) |
| accept length | 2.37 / 5 | 2.77 / 5 |
| GPU 使用 | 只 verify | draft + verify 都占 |

按 prompt 分组（CPU multi-round）：
- 代码类（binary_search / Rust / JSON 生成）：**80-88%**，很高
- 结构化输出 / 简单 Q&A：40-60%
- 开放长文 / 创作：12-28%

这个分布和 STANDALONE 完全一致，证明**差异在模型层面，和跑哪里（CPU vs GPU）无关**。

## 理论吞吐估算

前提数字（都是实测）：
- CPU draft **稳态** gen 速度：90 tok/s（长输出 / KV 连续情况下，见 llama.cpp 直接测）
- GPU target 单步 verify：~30 ms
- 每轮 CPU draft 产 5 token：5/90 ≈ **56 ms**

### 三档速度

| 场景 | iter_time | tokens/iter | tok/s |
|---|---|---|---|
| 本 demo 实测 (reset 每轮) | 112 ms + 30 ms | 3.37 | 24 |
| steady (KV 保留) | 56 ms + 30 ms | 3.37 | **39** |
| overlap (CPU/GPU 并行) | max(56, 30) = 56 ms | 3.37 | **60** |

### 和现有方案的对比（真实数据上）

| 方案 | tok/s | 说明 |
|---|---|---|
| baseline 无投机 | 36 | 基线 |
| STANDALONE (sglang) | **38.7** | 几乎没有提速，原始 65 是模板数据上的 |
| NGRAM (sglang) | ~60 * | 真实数据上估算，待单独测；之前的 300 是模板数据 |
| **CPU draft overlap 上限** | **60** | 超过 STANDALONE，持平 NGRAM |

*NGRAM 在真实数据上的真实数字未测，建议接下来补。

## 结论修正

原结论（已废弃）：
> ~~"瓶颈是 draft 模型本身不匹配 target，需要训练 GLM-4 专属 draft"~~

实际结论：
1. **draft 模型完全能胜任** —— accept rate 47%，和 STANDALONE 一致
2. **CPU 本身不是瓶颈** —— AMX 让 0.6B 模型跑 90 tok/s
3. **瓶颈在当前 demo 的实现方式**：每轮都 reset + 重 tokenize + 重 prompt eval，把
   每轮时间从 56ms 拉到 112ms，丢掉了一半速度
4. **真正的收益要靠两件事**：
   - a) 维护 draft 的 KV 缓存（不是每次 reset）
   - b) CPU 和 GPU 并行（draft 下一轮时 GPU 在 verify 这一轮）

两者都在 sglang 集成里天然支持（`StandaloneWorker` 就是这么做的，只是 worker 放在 GPU）。

## 下一步建议

### 优先级 1：把 NGRAM 在真实数据上的数字测出来
目前"NGRAM 300 tok/s"来自模板化数据。真实数据上大概率会掉到 50-100 tok/s。
这个数字决定 CPU draft 方案最终要超的真正门槛。

### 优先级 2：真正的 sglang 集成（Phase 1 步骤 3）
按 `docs/03_cpu_draft_poc_plan.md` 的方案 A：
- 继承 `StandaloneWorker`，派生 `CPUStandaloneWorker`
- 把 draft model 的 forward 改到 CPU（用 llama.cpp 的 C++ 接口或 PyTorch CPU）
- 维持 draft KV 缓存跨轮不 reset
- 实现 CPU draft / GPU verify 的 async overlap
- 目标：达到 demo 估算的 60 tok/s

### 优先级 3：更大更好的 draft 模型
既然准确度不是瓶颈，可以考虑放大到 1-2B 的 draft 换更高 accept rate：
- 如果 accept rate 能从 47% 提到 70%（accept_len 3.5），overlap throughput 可到 (3.5+1) / 56ms = **80 tok/s**
- 但更大的 draft 模型 CPU 跑会变慢，需要权衡

## 生成的数据文件
- `outputs/cpu_draft_demo/cpu_draft_demo_20260423_163329/` — v2 单轮 Q4_0（accept 17%）
- `outputs/cpu_draft_demo/cpu_draft_F16_20260423_170348/` — v2 单轮 F16（accept 7%）
- `outputs/cpu_draft_demo/cpu_draft_F16_multiround_20260423_171928/` — **多轮 F16（accept 47%，关键数据）**
- `outputs/standalone_realistic/standalone_realistic_20260423_165953/` — STANDALONE 对照
