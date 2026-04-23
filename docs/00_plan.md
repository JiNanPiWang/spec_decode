# 项目计划：自研推理在 GLM 上超过 sglang 原生投机采样

> 背景：可能要和 GLM 团队（智谱）合作做推理承接，老板的验证点是"在 GLM 上的投机采样吞吐超过 sglang 原生实现"。老板给的具体方向：**draft 模型放 CPU 上跑，GPU 只做 target verify**。

## 0. 范围和阶段划分

| 阶段 | 目标 | 交付物 |
|---|---|---|
| Phase 0 | 跑通 + 弄懂 sglang | 基线 tok/s 三行表、sglang specdec 梳理文档 |
| Phase 1 | CPU draft 可行性验证（老板点名方向） | 硬约束验证结论 + 最小原型 |
| Phase 2 | 如 Phase 1 通过：流水线优化 + 集群方案 | 完整方案 + 集群基线 |

当前只做 Phase 0 和 Phase 1。完整 benchmark matrix 留到 Phase 2。

## 1. 环境和硬件

- **当前**：单张 RTX 5090，模型 GLM-4-9B int4 量化版
- **未来**：8 卡 / 16 卡 A100 集群，跑全量 GLM5
- 约束：先在小服务器上跑通，再上集群

## 2. Phase 0 要做的事（约 1 周）

### 2.1 跑通三个 server 变体
脚本见 `launch_sglang.sh`：
1. **baseline**：sglang 裸跑 GLM4-int4，无投机
2. **ngram**：sglang 自带的 NGRAM 投机采样（无需 draft 模型）
3. **standalone / eagle**：如果有合适的 draft 模型 / EAGLE 权重再跑

> EAGLE 需要和 target 配套的 draft 权重；GLM4 如果没有官方 EAGLE 权重，先跳过，ngram 能跑就行。

### 2.2 跑单点 benchmark
脚本见 `bench_phase0.py`。**只测一种配置**（batch=1, input≈512 chars, output=128 tokens），得到一张三行表：

| 变体 | tok/s | avg latency | P99 latency |
|---|---|---|---|
| baseline | ... | ... | ... |
| sglang ngram | ... | ... | ... |
| 我们的 ngram | ... | ... | ... |

**不要**在这一步就横铺 batch × seq matrix。那是 Phase 2 的事。

### 2.3 读源码，形成对 sglang specdec 的理解
参考 `02_sglang_analysis.md`。要能讲清楚：
- sglang 支持哪几种投机采样算法
- draft 和 verify 是怎么串在一起的（`forward_batch_generation` → `draft` → `verify`）
- 每种算法的代价和适用场景

### 2.4 数据分布警示
我们现有 ngram 方案在测试数据上表现很快，但这**可能来自数据结构相似**（重复/模板化）。合作方复测时如果换数据，数字会塌。
行动项：Phase 0 主表用**当前测试数据**；另外单独跑一份"非模板化数据"（开放问答、创作），**不进主表但留着**，汇报时必须带上。

## 3. Phase 1：CPU draft POC（约 1–2 周）

详见 `03_cpu_draft_poc_plan.md`。核心是先验一条**硬约束**：

```
draft(CPU) 产出速率 × 平均接受长度 > target(GPU) 纯解码速率
```

不过这条，CPU 方案就是负收益，直接写报告终止 —— 老板要的是"评估代价"，这个结论本身就是有价值的交付。

过了这条，再做最小原型：基于 sglang 的 `StandaloneWorker`，把 draft 模型 forward 迁到 CPU。

## 4. 给老板的交付

Phase 0 + Phase 1 结束后，一页纸报告回答四个问题：
1. sglang 在 GLM4-int4 上的基线 tok/s
2. CPU draft 硬约束有没有过
3. 如果继续做完整实现，工程量估计
4. 其他可选方向（EAGLE-3 / REST / Medusa）的 ROI 排序

## 5. 下一阶段要和老板对齐的问题

1. GLM4-int4 是官方量化还是我们自己量化？（会影响和智谱原生推理栈的对比）
2. 合作方的 baseline 是 sglang 还是他们自己的栈？如果是后者，Phase 0 的对照组要换
3. 目标场景是低延迟（batch=1）还是高吞吐（大 batch）？**投机采样在大 batch 下收益会急剧下降**
4. GLM5 在集群上的并行策略（TP/PP/EP）—— 单机 CPU draft 方案可能不直接适用于多卡
