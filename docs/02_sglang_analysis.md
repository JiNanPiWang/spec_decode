# sglang 投机采样源码分析

> 目标读者：已经看过 `01_specdec_intro.md`，对 specdec 原理有基本概念。
> 这篇讲 sglang 的**代码组织**和**五种算法的差异**。

> 源码路径：`sglang/python/sglang/srt/speculative/`

## 1. 骨架：四个抽象概念

sglang 把投机采样拆成四个角色：

| 角色 | 说明 | 文件 |
|---|---|---|
| `SpeculativeAlgorithm` | 算法枚举（NONE / EAGLE / EAGLE3 / STANDALONE / NGRAM / DFLASH） | `spec_info.py` |
| `BaseSpecWorker` | 抽象基类：有 `target_worker` 和 `draft_worker` | `base_spec_worker.py` |
| `SpecInput` / `SpecInputType` | draft/verify 阶段之间传递的数据结构（draft tokens、tree mask、retrieve index 等） | `spec_info.py` |
| 各算法的 Worker | 每种算法一个具体 Worker 类 | `eagle_worker.py` / `ngram_worker.py` / ... |

每个算法都要同时提供：
1. **怎么产 draft**（`draft()`）
2. **怎么做 verify**（`verify()` or `prepare_for_verify()`）
3. **怎么串 draft + verify**（`forward_batch_generation()`）

## 2. sglang 支持的五种算法总览

从 `spec_info.py` 的枚举看：

```python
class SpeculativeAlgorithm(Enum):
    DFLASH = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()
```

对应的 worker：

| 算法 | 主要 worker 类 | 第二套（overlap 调度） |
|---|---|---|
| `EAGLE` / `EAGLE3` | `EAGLEWorker` | `EAGLEWorkerV2` |
| `EAGLE` + `--enable-multi-layer-eagle` | `MultiLayerEagleWorker` | `MultiLayerEagleWorkerV2` |
| `STANDALONE` | `StandaloneWorker`（继承自 `EAGLEWorker`） | `StandaloneWorkerV2` |
| `NGRAM` | `NGRAMWorker` | （不支持 overlap） |
| `DFLASH` | `DFlashWorker` | （不支持 overlap） |

"V2 / overlap" 是指 draft 和 verify 能异步重叠，代码更复杂但吞吐更高。Phase 0 读 V1 版本就够。

下面对每种算法做简要分析。

---

## 3. NGRAM：查表式 draft

**核心思想**：不用模型，用 **n-gram trie** 从历史 token 里匹配后续。

**适用**：RAG、代码、结构化输出 —— 输出里经常"抄"输入的场景。

**文件**：`ngram_worker.py` + `ngram_info.py` + `cpp_ngram/` (C++ 加速的 trie)

**关键数据结构**：
- `NgramCorpus`：一个全局 trie，持续吸收历史 token 序列
- 每个请求的 `origin_input_ids + output_ids` 尾部会查表，拿到候选延续

**关键 CLI 参数**：

```
--speculative-algorithm NGRAM
--speculative-num-draft-tokens 16        # 每次 verify 塞多少 draft（树总节点数）
--speculative-ngram-max-trie-depth 18    # trie 最大深度（匹配的最长 n-gram）
--speculative-ngram-capacity 10000000    # trie 容量（总 token 数）
--speculative-ngram-min-bfs-breadth 1    # 每层 BFS 最少几个分支
--speculative-ngram-max-bfs-breadth 10   # 每层 BFS 最多几个分支
--speculative-ngram-external-corpus-path ...  # 外部语料冷启动（重要！）
```

最后一项很关键：**可以预加载外部语料**到 trie，避免冷启动 acceptance 低。

**核心流程**（简化自 `ngram_worker.py`）：

```python
class NGRAMWorker:
    def forward_batch_generation(self, batch):
        # 1. 从 trie 里对每个请求查出 draft tokens 和 tree mask
        self._prepare_for_speculative_decoding(batch)
        #    -> req_drafts, mask = self.ngram_corpus.batch_get(...)
        #    -> 构造 tree_mask, retrieve_index, positions
        #    -> batch.spec_info = NgramVerifyInput(...)
        #    -> batch.forward_mode = TARGET_VERIFY

        # 2. target 模型一次 forward 验证整棵树
        result = self.target_worker.forward_batch_generation(...)

        # 3. 把这一轮产出的 token 塞回 trie 供下次用
        self._update_ngram_corpus(batch)
        return result
```

**没有独立的 "draft forward"** —— 因为 draft 就是查表。这是它为什么便宜。

---

## 4. EAGLE：专用 draft 网络

**核心思想**：训练一个小 network（通常 1 层 transformer decoder），输入是 target 模型的**上一层 hidden state + embedding**，输出是候选 draft tokens 的 logits。树状 draft + tree attention。

**EAGLE-2** 是 EAGLE 的升级版，做了动态 draft tree（更 context-aware）。
**EAGLE-3** 进一步解耦了对 target hidden state 的依赖，泛化性更好。

sglang 里 EAGLE-2 / EAGLE-3 共享 `EAGLEWorker`，只在数据流上有些 switch（`hot_token_id` / `SpeculativeAlgorithm.is_eagle3()`）。

**文件**：
- `eagle_worker.py`（主流程）
- `eagle_info.py`（`EagleDraftInput` / `EagleVerifyInput` / `EagleVerifyOutput`）
- `eagle_utils.py`（树构造：`build_tree_kernel_efficient`）
- `eagle_draft_cuda_graph_runner.py`（draft 阶段的 CUDA graph）

**关键 CLI 参数**：

```
--speculative-algorithm EAGLE            # 或 EAGLE3
--speculative-draft-model-path ...       # draft 网络权重
--speculative-num-steps 5                # draft 树的深度
--speculative-eagle-topk 4               # 每层保留的 top-k 分支
--speculative-num-draft-tokens 8         # 最终 verify 时树的节点总数
--speculative-token-map ...              # （EAGLE 可选）热词表，缩小 draft vocab
```

**核心流程**（`eagle_worker.py:432` `forward_batch_generation`）：

```python
def forward_batch_generation(self, batch):
    if batch.forward_mode.is_extend():
        # prefill 阶段：target 先 forward，然后 draft 网络 extend 一次
        logits, next_ids, ... = self.forward_target_extend(batch)
        self.forward_draft_extend(batch, logits.hidden_states, next_ids, ...)
        return GenerationBatchResult(...)
    else:
        # decode 阶段：draft → verify → draft_extend
        spec_info = self.draft(batch)                    # 产 draft 树
        logits, verify_out, ... = self.verify(batch, spec_info)  # target 一次 forward 验证
        self.forward_draft_extend_after_decode(batch)    # 把本轮新 token 喂给 draft 网络
        return GenerationBatchResult(...,
                 num_accepted_tokens=sum(verify_out.accept_length_per_req_cpu))
```

`draft()` 里做的是：

```python
def draft(self, batch):
    self._draft_preprocess_decode(batch)          # 准备 draft 的输入
    # 在 draft model 上跑 num_steps 层树状展开
    parent_list, top_scores_index, draft_tokens = self.draft_forward(forward_batch)
    # 把展开结果组装成 (tree_mask, position, retrieve_index, ...) 给 verify 用
    tree_mask, pos, retrieve_index, retrieve_next_token, retrieve_next_sibling, draft_tokens = \
        build_tree_kernel_efficient(verified_id, parent_list, top_scores_index, draft_tokens,
                                     seq_lens, seq_lens_sum,
                                     self.topk, self.speculative_num_steps,
                                     self.speculative_num_draft_tokens)
    return EagleVerifyInput(...)
```

`verify()` 里做的是：

```python
def verify(self, batch, spec_info):
    spec_info.prepare_for_verify(batch, self.page_size)  # 准备 KV cache slot 等
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    # target 一次 forward，同时拿到树上所有位置的 logits
    batch_result = self.target_worker.forward_batch_generation(..., is_verify=True)
    # 接受-拒绝协议，挑最长被接受的路径
    res = spec_info.verify(batch, logits_output, kv_pool, ...)
    # 回滚未接受 token 对应的 KV cache，准备下一轮
    batch.spec_info = res.draft_input  # 传给下一次 draft
    return logits_output, res, ...
```

树的构造、tree mask、retrieve_index 这些细节在 `eagle_utils.py:build_tree_kernel_efficient`，C++/CUDA kernel 实现，这里不展开，理解它"把并列分支展平成一条序列 + 一个 mask"就够了。

### Multi-layer EAGLE（`multi_layer_eagle_*.py`）
是 EAGLE 的堆叠版本，draft 网络本身是多层的。触发：`--enable-multi-layer-eagle`。Phase 0 不用管。

---

## 5. STANDALONE：独立小模型做 draft

**核心思想**：不训练专用 draft 头，直接用一个和 target 同系列的**完整小模型**做 draft（比如 70B target 配 7B draft）。

**和我们老板要做的"CPU draft"最相关的就是这条路径**。

**文件**：`standalone_worker.py`（185 行，非常薄）

**看代码关键点**：

```python
class StandaloneWorker(EAGLEWorker):
    # 注意：直接继承 EAGLEWorker！
    def __init__(self, server_args, ...):
        # 设置参数、加载模型
        # 关键：在 TpModelWorker.__init__(..., is_draft_worker=True, ...) 里
        # 会读 --speculative-draft-model-path，把它作为独立模型加载
        ...
        TpModelWorker.__init__(
            self,
            ...,
            is_draft_worker=True,
            req_to_token_pool=self.req_to_token_pool,          # 和 target 共享请求池
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,  # 共享 KV pool allocator
            ...
        )
```

**为什么继承 EAGLEWorker？** 因为 draft→verify 的外层逻辑（tree 构造、verify 协议、CUDA graph 管理）完全复用。Standalone 只是换了个 draft 网络的 forward 实现。

**关键 CLI 参数**（和 EAGLE 一样，但不需要 token map）：

```
--speculative-algorithm STANDALONE
--speculative-draft-model-path <small_glm>
--speculative-num-steps 3
--speculative-eagle-topk 4
--speculative-num-draft-tokens 8
--speculative-draft-model-quantization int4   # draft 自己的量化
```

**对我们 CPU 方案的启示**：
老板说的"CPU draft"其实就是 STANDALONE 的一个变种 —— draft 模型的 forward 跑在 CPU 上。改动点不大（理论上只要在 draft model runner 初始化时换 device），但 CUDA graph / attention backend 都要绕开，实际工作量见 `03_cpu_draft_poc_plan.md`。

---

## 6. DFLASH：块并行 draft

**文件**：`dflash_worker.py`（1256 行，新算法，比较重）

**核心思想**（根据源码推测）：用一个 **mask token + draft window** 的机制，在 target 自己身上做块并行解码。类似 diffusion-style / block-parallel decoding。

**不支持 overlap scheduling**（见 `spec_info.py`：`if self.is_dflash(): if enable_overlap: raise`）。

**对我们优先级低**：是一个独立研究方向，想用需要额外训练/改造，不是即插即用。Phase 0 不测。

---

## 7. 串起来：scheduler 是怎么调用的

入口在 `python/sglang/srt/managers/scheduler.py`（或相近位置）的 `run_batch` 里。简化后：

```python
# 伪代码
if spec_worker is None:
    # 无投机采样
    result = tp_worker.forward_batch_generation(batch)
else:
    # 有投机采样
    result = spec_worker.forward_batch_generation(batch)
    # spec_worker 内部会调 tp_worker（=target_worker）
```

`forward_batch_generation` 返回 `GenerationBatchResult`，里面有个字段是 `num_accepted_tokens` —— 这就是统计 acceptance 用的。如果你想看实时 acceptance rate，抓这个字段的日志即可。

## 8. 对 Phase 0 的含义

读完这一章，三件事：

1. **我们 Phase 0 要跑通的 sglang 变体**：`baseline`（NONE）、`ngram`（NGRAM）、可选 `standalone`（如果能找到小 draft）。EAGLE 需要配套训练权重，GLM4 如果没有，暂不搞。
2. **在读 benchmark 结果时**：关注 acceptance rate（日志里的 `num_accepted_tokens / num_verify_steps`）。acceptance 低说明 draft 方法不适合数据。
3. **老板的 CPU draft 方案在 sglang 里对应的是 STANDALONE 路径**。要改的核心文件是 `standalone_worker.py` + `TpModelWorker` 里处理 `is_draft_worker=True` 的那部分（设备放置）+ 对应的 attention backend（CPU 版本）+ 绕开 CUDA graph。细节见下一篇。
