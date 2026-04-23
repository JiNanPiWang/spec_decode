# CPU Draft POC 计划（方案 A）

> 老板指定方向：draft 模型跑在 CPU 上，GPU 只做 target verify。
> 先做方案 A：**外挂 draft server**（独立进程，进程间通信）。
> 方案 B（直接改 sglang）放到后续，如果 A 证明可行再深入。

## 1. 先过硬约束：这个方向能不能做

投机采样收益的必要条件：

```
draft 产出速率 × 平均接受长度  >  target 纯解码速率
```

举例（数字用占位，Phase 0 会测真值）：
- target 在 5090 上 GLM4-int4 decode = 60 tok/s
- 假设接受长度 3 个 token
- 则 draft 在 CPU 上必须 ≥ 60 / 3 = **20 tok/s**
- 对一个 0.5B int4 模型在现代 CPU 上，这个量级可以（llama.cpp 实测 30–80 tok/s）
- 但如果 target 上 A100 集群 = 200 tok/s，CPU draft 就要 ≥ 67 tok/s，边缘了

**Phase 1 第一步：不要写集成代码，先在 CPU 上裸跑 draft 模型测 tok/s。** 这一步占用 1–2 天，不过就直接终止方案。

### 约束之外，还要看的事
1. **draft 的 accept rate**：CPU 跑更小模型 → 可能 accept rate 更低 → 每次接受长度变短 → 不等式收紧
2. **通信开销**：进程间传 token（每次几十字节）延迟比想象中大，在 batch=1 低延迟场景可能是 1–5ms / 次
3. **draft 和 verify 重叠可能性**：理想情况下 GPU 在 verify 当前一轮时 CPU 在准备下一轮 draft。这是方案 A 必须做到的，否则 CPU draft 毫无意义

## 2. 方案 A 架构

```
┌────────────────────────────┐          ┌─────────────────────────┐
│  sglang server（GPU）       │          │ CPU draft server        │
│  ─────────────              │          │ ─────────────           │
│  TargetWorker (GLM4 int4)   │   SHM/   │ llama.cpp / OpenVINO    │
│  CustomDraftWorker ─────────┼─ gRPC ──→│ 小模型 (0.5B int4)      │
│    └ forward_batch_gen()    │←─draft──│ KV cache (CPU)          │
│       ├ call remote draft   │          │                         │
│       ├ build tree          │          └─────────────────────────┘
│       └ verify on GPU       │
└────────────────────────────┘
```

**CustomDraftWorker** 是一个继承自 sglang `BaseDraftWorker` 的类，它的 `draft()` 方法不再调本地 GPU，而是发 RPC 到 CPU server。

**关键设计决定**：
1. **通信协议**：shared memory（最快，单机）还是 gRPC（标准，更灵活）？先用 **UNIX domain socket + protobuf 或 msgpack**，简单且延迟 <1ms
2. **draft 是否维护自己的 KV cache**：是。CPU 端完全独立维护自己的 KV，只有 token id 在进程间走
3. **如何 rollback**：当 target 拒绝 draft 的一部分 token 时，CPU 端也要 rollback 自己的 KV cache。需要一个明确的 `accepted_prefix_len` 回传协议

## 3. 实施步骤

### 步骤 1：CPU 侧独立能跑（2–3 天）
- 选一个候选 draft（建议先试 Qwen2.5-0.5B 或 GLM 系列最小版，注意 tokenizer 对齐）
- 用 llama.cpp 量化到 Q4_K_M 或 INT4
- 写一个 Python 封装，暴露 `draft(context_ids: list[int], n_steps: int) -> list[list[int]]` 接口
- **裸测 tok/s 和单次 forward latency**
- 判决：过 / 不过硬约束

**如果不过，报告终止，进入 Phase 2 的其他方向（EAGLE-3 / REST / Medusa）**。

### 步骤 2：最小 RPC server（1–2 天）
- `cpu_draft_server.py`：启动 draft model，监听 UDS
- 协议：
  ```
  request: { request_id, context_tokens, n_steps, n_topk }
  response: { draft_tree_tokens, parent_indices, draft_probs }
  ```
- 客户端 stub：`cpu_draft_client.py`

### 步骤 3：sglang 侧接入（3–5 天）
- 派生一个 `CPUStandaloneWorker`，继承或基于 `StandaloneWorker`
- 替换 `draft()` 里调本地 draft model 的部分为调 `cpu_draft_client`
- 关键：**保留 sglang 原本的 tree 构造和 verify 逻辑**，只把 draft forward 替换
- 绕开 CUDA graph（draft 部分不再在 GPU）

### 步骤 4：重叠优化（3–5 天）
- 让 CPU server 在 verify 进行时就开始下一轮 draft（预测性）
- 实现方式：verify 发出时，同时给 CPU 发一个"投机性 continue"消息，让 CPU 基于**最可能的接受路径**继续生成；如果最终被拒，CPU 端 rollback
- 这一步是方案 A 真正的性能来源

### 步骤 5：单点 benchmark（1 天）
- 用 `bench_phase0.py` 跑同一个 label
- 对比：baseline / sglang ngram / sglang standalone / **CPU draft**

## 4. 风险和退出条件

| 风险 | 退出判据 |
|---|---|
| 硬约束不过 | 步骤 1 后终止 |
| CPU draft accept rate < 30% | 考虑换更大 draft 或终止 |
| RPC overhead > 2ms | 切方案 B（in-process 改 sglang） |
| 重叠优化做不出 10%+ 收益 | 说明架构上不成立，终止 |

## 5. 工作量估计

| 步骤 | 乐观 | 悲观 |
|---|---|---|
| 步骤 1 CPU 裸测 | 2 天 | 3 天 |
| 步骤 2 RPC server | 1 天 | 2 天 |
| 步骤 3 sglang 接入 | 3 天 | 5 天 |
| 步骤 4 重叠优化 | 3 天 | 5 天 |
| 步骤 5 bench + 报告 | 1 天 | 2 天 |
| **合计** | **10 天** | **17 天** |

实际按 2–3 周规划，中间必然会踩 sglang 内部约定的坑（比如 KV 共享、req_to_token pool、page_size 对齐等）。

## 6. 和"直接改 sglang（方案 B）"的对比

| 维度 | 方案 A（外挂 server） | 方案 B（改 sglang 内部） |
|---|---|---|
| 上手成本 | 低 | 高（要读 TpModelWorker） |
| 通信开销 | RPC：~1ms/次 | 进程内：微秒级 |
| 调试难度 | 容易（两个进程分别 debug） | 难（CUDA/CPU 混布，graph 要绕开） |
| 性能上限 | 受 RPC 限 | 高 |
| 给合作方移植 | 容易（解耦） | 难（绑 sglang fork） |

**推荐先方案 A 打通**，拿到性能数据，再根据数据决定是否做方案 B。

## 7. 开工前的 checklist

- [ ] 和老板对齐：Phase 0 结果达到什么标准才进 Phase 1（比如 sglang ngram 已经打得很好时，还做不做 CPU draft？）
- [ ] GLM4 的 tokenizer 能否和候选 draft 模型对齐？（不对齐工作量翻倍）
- [ ] 5090 宿主 CPU 型号和核数？（决定 draft 上限）
- [ ] 合作方业务场景确认：batch=1 低延迟 vs 大 batch 高吞吐
