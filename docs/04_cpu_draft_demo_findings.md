# CPU Draft Demo 结果汇总（2026-04-23 / 04-24，v3）

> **版本说明**：本文档在 2026-04-23 同一天经过两次重要更新，又在 2026-04-24 补测 NGRAM 基线：
> - 最早（v1）得出"CPU draft 和 target 不匹配，根本不可用"的**错误**结论
> - 多轮测试（multi-round）重新评估，CPU draft accept rate 约 47%，**和 sglang
>   STANDALONE 同级别**，原来的 17% 是"单轮测试"的系统性偏差
> - **v3 (2026-04-24)**：补测 NGRAM 在同一套 realistic prompt 上的数字，把之前记忆里的
>   "NGRAM 228 tok/s ≈ 5x" 更新为 **真实数字 89.4 tok/s (median)**；原来的 228 是
>   sglang 自带 benchmark_serving 的模板化语料，prompt 高度重复被 n-gram 命中拉高

## 一句话结论

**CPU draft + GPU target 方向上可行**：accept rate 47%，和 sglang STANDALONE 在相同
模型对、相同 realistic 语料上测出的 46% 基本一致。理论最优（KV 缓存保留 + CPU/GPU
并行）吞吐约 **60 tok/s**，**超过** STANDALONE 在真实语料上的 38.7 tok/s，
**但低于 NGRAM 实测 89.4 tok/s**（同一 realistic prompt 套件，2026-04-24 补测，见
`outputs/ngram_realistic/`）。

要超越 NGRAM 需要 **Phase 1 集成 + 提升 accept_len** 两个动作一起做，单独做 Phase 1 的
overlap 只能拿到 ~60 tok/s 这个估算值。

## 方法

### 第一轮：单轮测试（cpu_draft_demo.py v2）
- accept rate 16.7%，据此推测"`GLM-4.5-DRAFT-0.6B` 不适合给 `GLM-4` 当 draft"
- 对比 F16 draft，accept rate 还更低（6.7%），说明问题不在量化

### 第二轮：参考 sglang STANDALONE（bench_standalone_realistic.py）
- 同一组模型、同一组 prompt、同一个 target
- **STANDALONE 自己打出来 accept rate 才 40% (median) / 46% (mean)**
- 2.4× 的差距不合理

### 第三轮：side-by-side 文本对比
以 prompt `def binary_search(arr, target):` 为例：
```
TARGET : ' \n    low = 0\n    high ='
DRAFT  : ' \n    left = 0\n    right ='
```
两个模型都写了正确的 binary_search，只是一个用 `low/high`，一个用 `left/right` —— 第一个
token 就不相邻，之后所有 token 在单轮里都记"不匹配"。

### 第四轮：多轮测试（cpu_draft_demo_multiround.py）
模拟真正的 spec decode 行为：draft 出错之后，把 target 的正确 token 并入 context，下一轮
继续。这才匹配 sglang STANDALONE 的内部逻辑。
- **120 轮测试，overall accept rate 47.3%**
- 和 STANDALONE 的 46.3% 基本一致

### 第五轮：NGRAM realistic 基线（2026-04-24 补测）
用同一个 realistic prompt 套件打 sglang NGRAM（同 target、同硬件）：
- **accept rate 30.5% (median) / 28.7% (mean)**（相对 `num_draft_tokens=8`）
- **accept_len 2.44 / 8**（和 CPU draft 的 2.37 / 5 几乎等价的 per-token 接受质量）
- **gen throughput 89.4 tok/s (median)**

比之前记的 228 tok/s 低了 60%。原因：sglang 的默认 benchmark_serving 语料有大量
重复 prompt 和结构，n-gram trie 命中率接近满值；在 12 个独立长 prompt 下退化到
30% 左右。

## 核心数字（5090 + Xeon 8470Q，12 个真实长 prompt）

### accept 质量（和 STANDALONE / NGRAM 横向对比）

| 方案 | accept rate | accept_len / max draft tokens | 对比 |
|---|---|---|---|
| CPU draft multi-round | **47.3%** | 2.37 / 5 | 基线 |
| sglang STANDALONE | 46.3% | 2.77 / 5 | 和 CPU draft 基本一致 |
| sglang NGRAM (realistic) | 30.5% | 2.44 / 8 | per-token 质量也相当，只是 draft tokens 更多 |

结论：**draft 模型的接受质量在 realistic prompt 上和 NGRAM/STANDALONE 处于同一量级**。

### 端到端吞吐（tok/s，都是 median）

| 方法 | tok/s | 说明 |
|---|---|---|
| baseline 无投机 | ~36 | 纯 target 解码 |
| STANDALONE (sglang, GPU draft 0.6B) | **38.7** | 基本没加速，draft 挤占 GPU |
| **NGRAM (sglang)** | **89.4** | 同套 realistic prompt 实测，是目前最强基线 |
| CPU draft 当前 demo（每轮 reset） | ~24 | 实测，受 demo 实现里每轮重 tokenize 拖累 |
| CPU draft steady（KV cache 保留） | ~39 | 估算，去掉 prompt eval 开销 |
| CPU draft overlap（+CPU/GPU 并行） | **~60** | 估算，上限 |

**CPU draft overlap 估算 60 tok/s：超过 STANDALONE，但仍低于 NGRAM ~30%**。

### 底层时序（CPU 单路估算，供汇报用）

- CPU draft 稳态 gen 速度：~90 tok/s（0.6B F16 on AMX，去掉 prompt eval）
- CPU draft 5 token：5/90 ≈ **56 ms**
- GPU target verify 5 token：~30 ms（按 STANDALONE 数据外推）
- overlap iter_time = max(56, 30) = 56 ms，每 iter 产出 3.37 token → **60 tok/s**

对比 NGRAM：draft 从 n-gram trie 查询近乎 0 成本，iter_time 由 target verify 8 token
主导约 38 ms，产出 3.44 token → 89 tok/s。**CPU draft 要打平 NGRAM，必须让 draft cost
降到 <30 ms，或者用更准的 draft 把 accept_len 从 2.37 拉到 3.5+**。

## 结论讨论

v1 原结论（已废弃）：
> ~~"瓶颈是 draft 模型本身不匹配 target，需要训 GLM-4 专用 draft"~~

当前结论：
1. **draft 模型完全能胜任** —— accept rate 47%，和 STANDALONE 一致，和 NGRAM 的
   per-token accept 质量也基本持平
2. **CPU 不是数字瓶颈** —— AMX 跑 0.6B F16 稳态 90 tok/s，是可以 hide 在 GPU verify
   后面的
3. **当前 demo 的拖慢**是实现问题 —— 每轮 reset + 重 tokenize + 重 prompt eval，
   每轮额外 56 ms；要 steady 到 39 tok/s 需要修掉
4. **要真正用起来需要两件事**：
   - a) 维护 draft 侧 KV 缓存（不再每轮 reset）
   - b) CPU 和 GPU 并行（draft 下一轮时 GPU verify 当前轮）

这两件事在 sglang 架构里天然支持（`StandaloneWorker` 就是这么做的，只是 worker 挂在 GPU 上）。

## 下一步规划

### ~~优先级 1：补 NGRAM 在真实工作上的数字测量~~（已完成 2026-04-24）
实测 NGRAM 在 realistic 12 prompt 上 gen throughput **89.4 tok/s (median)**，
accept rate 30.5% / accept_len 2.44（对 8 draft token）。原来记的 228 tok/s
是 sglang 内置 benchmark 的模板化语料，有大量 n-gram 命中；在独立长 prompt 下
下降到原来的 39%。

**这意味着真正要超越的目标就是 ~89 tok/s**，不是之前担心的 228。
当前 CPU draft overlap 估算 60 tok/s 相对差距约 30%，不算大但也不是打平。

### 优先级 1（new）：Phase 1 集成 + 提升 accept rate
要超越 NGRAM，两条腿：

- **a) Phase 1 集成到 sglang**（见 `docs/03_cpu_draft_poc_plan.md` 方案 A），
  把 56 ms/iter 的 overlap 估算变成实测，验证 60 tok/s 这个数字。
- **b) 提升 accept_len 到 3.0+**：accept_len 每上 0.3，overlap throughput 多约 6 tok/s。
  路径：(i) 把 draft 换成更准的（1-2B），(ii) 针对 GLM-4 做一版 finetune/EAGLE 头。

单 (a) 只能拿到 60 tok/s，差 NGRAM ~30%；(a)+(b) 到 accept_len 3.5 时，
overlap 估算 (3.5+1)/56ms ≈ 80 tok/s，接近 NGRAM。要明确胜出 NGRAM 还得加
(c) CPU draft 本身加速（更小 draft、量化到 Q4、n_draft 调优）。

### 优先级 2：完成 sglang 集成 Phase 1
按 `docs/03_cpu_draft_poc_plan.md` 方案 A：
- 继承 `StandaloneWorker`，做一个 `CPUStandaloneWorker`
- 把 draft model 的 forward 改到 CPU（接 llama.cpp 的 C++ 接口或 PyTorch CPU）
- 维持 draft KV 缓存，不再每轮 reset
- 实现 CPU draft / GPU verify 的 async overlap
- 目标：达到 demo 估算的 60 tok/s

### 优先级 3：尝试更好的 draft 模型
既然准确度才是瓶颈，可以考虑放大 1-2B 的 draft 来抬高 accept rate：
- 如果 accept rate 能从 47% 提到 70%，accept_len 3.5，overlap throughput 可达
  (3.5+1) / 56ms ≈ **80 tok/s**
- 但更大的 draft 在 CPU 上会更慢，需要权衡

## 相关的输出文件
- `outputs/cpu_draft_demo/cpu_draft_demo_20260423_163329/` — v2 单轮 Q4_0（accept 17%）
- `outputs/cpu_draft_demo/cpu_draft_F16_20260423_170348/` — v2 单轮 F16（accept 7%）
- `outputs/cpu_draft_demo/cpu_draft_F16_multiround_20260423_171928/` — **多轮 F16（accept 47%，关键数据）**
- `outputs/standalone_realistic/standalone_realistic_20260423_165953/` — STANDALONE 基线
- `outputs/ngram_realistic/ngram_realistic_20260424_112724/` — **NGRAM 基线（同套 prompt，2026-04-24 补测）**
