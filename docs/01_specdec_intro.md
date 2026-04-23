# 投机采样入门：原理和直觉

> 目标读者：没接触过投机采样（speculative decoding）的工程师。
> 看完这篇能回答："为什么这玩意能加速？它为什么不会影响生成质量？"

## 1. 为什么大模型解码慢

大模型自回归生成是**一次一个 token**：

```
输入 prompt → 模型 forward → 产出 token_1
prompt + token_1 → 模型 forward → 产出 token_2
prompt + token_1 + token_2 → 模型 forward → 产出 token_3
...
```

每一步都是一次完整的 forward。对 70B / 9B 这种规模的模型，一次 forward 在 GPU 上耗时 10–50 ms。

但关键是：**解码阶段 GPU 并没有打满算力，它是被显存带宽卡住的**。模型参数每个 token 都要从 HBM 读一遍，GPU 的 TFLOPS 根本没用上。

换句话说：如果我一次 forward 能"验证"好几个 token，成本几乎不变，GPU 就被更充分利用了。这就是投机采样的切入点。

## 2. 核心思路：先猜后验

```
1. 用一个便宜的机制（小模型 / n-gram / 查表）一次性"猜"出未来 K 个 token。
   这叫 draft（打草稿）。
2. 把这 K 个 token 一起塞进大模型做 forward。
   大模型一次 forward 同时产出 K+1 个位置的 logits。
   这叫 verify（核对）。
3. 从头比对：大模型在每个位置的"首选 token"是否和猜测一致？
   一致的接受，第一个不一致的位置用大模型自己的选择替换，后面的都丢。
```

关键洞察：**被接受的 token，其分布和大模型直接逐步采样完全等价**（有严格数学证明，叫 speculative sampling 的拒绝采样协议）。所以**生成质量没有任何损失**。

这是投机采样最漂亮的一点：不是"快但差一点"，是"快且严格等价"。

## 3. 一个最小直觉例子

假设大模型要生成 "The cat sat on the mat."

- **naive 解码**：7 次 forward（每个词一次）
- **投机采样**（假设 draft 猜 4 个）：
  - draft 猜："The cat sat on"（4 个 token）
  - 大模型一次 forward → 产出 5 个位置的 logits
    - 位置 1 首选："The" ✓
    - 位置 2 首选："cat" ✓
    - 位置 3 首选："sat" ✓
    - 位置 4 首选："on" ✓
    - 位置 5 首选："the" ← 大模型自己产的，白赚一个
  - 一次 forward 前进 **5 个 token**（4 个接受 + 1 个 bonus）
  - 如果后续继续全猜中，相当于把 7 次解码压缩成 2 次

实际 acceptance rate 大约 50%–80%，平均每次 verify 前进 2–4 个 token。理论加速 2–4x。

## 4. 代价：draft 必须"又快又准"

投机采样能赚的前提：

```
T_total(spec) = T_draft + T_verify < K × T_decode(naive)
其中 K 是平均被接受的 token 数
```

这意味着：
- **draft 必须够快**：如果 draft 本身跟 target 一样慢，就白忙
- **draft 必须够准**：accept rate 太低，每次猜 K 个但只过 1 个，draft 开销全浪费

常见的几类 draft 方法，就是在"快"和"准"之间做不同权衡：

| 方法 | 快？ | 准？ | 代价 |
|---|---|---|---|
| N-gram 查表 | 非常快 | 看数据，重复场景可以很准 | 只对有模式的数据有效 |
| 小模型（7B target 配 1B draft） | 中等 | 中等 | 要加载额外模型，吃显存 |
| EAGLE（专用 draft 头） | 快 | 高 | 要训练配套权重 |
| Medusa（多头并行预测） | 非常快 | 中等 | 要训练，只看 1 步未来 |

## 5. 树状 draft（tree attention）

更进一步：draft 不一定是一条线性序列，可以是一棵树。

```
             root
            /    \
        "the"    "a"
        /  \      |
    "cat" "dog" "car"
     /
   "sat"
```

大模型一次 forward 可以把整棵树都 verify 了（每个节点都拿到它在自己分支上的 logits），最后挑一条最长被接受的路径。

好处是：单次 draft 提供更多选择，accept rate 更高。

代价是：
- forward 的有效 token 数变大，verify 本身变慢
- 需要一种叫 **tree attention** 的魔改 attention mask（兄弟节点之间互相不可见，只看祖先）

sglang 的 EAGLE 和 NGRAM 都用树状 draft。

## 6. 一段玩具代码（非 sglang，只为看原理）

```python
def speculative_step(target_model, draft_model, context, k=4):
    # 1) draft: cheap K-step autoregression
    draft_tokens = []
    cur = context
    for _ in range(k):
        tok = draft_model.greedy_next(cur)
        draft_tokens.append(tok)
        cur = cur + [tok]

    # 2) verify: target does ONE forward on context + draft_tokens,
    # returning logits at K+1 positions.
    logits = target_model.forward(context + draft_tokens)  # shape: (K+1, vocab)

    # 3) accept-reject
    accepted = []
    for i, tok in enumerate(draft_tokens):
        tgt_choice = argmax(logits[i])
        if tgt_choice == tok:
            accepted.append(tok)
        else:
            # Target disagrees here: overwrite with target's pick, drop rest of draft
            accepted.append(tgt_choice)
            return accepted
    # All K drafts accepted: use target's K-th position as free bonus token
    accepted.append(argmax(logits[k]))
    return accepted
```

这段贪心版本就够用来建直觉了。真正的 sglang 实现要处理：温度采样（非 greedy）、tree mask、KV cache 回滚、分布式、CUDA graph—— 细节在 `02_sglang_analysis.md`。

## 7. 什么场景下投机采样收益最大

- **batch 小**：batch=1 时 GPU 严重 memory-bound，投机采样把计算填进去，收益最高
- **输出长**：输出越长，累积加速越明显
- **模式化数据**：如代码、RAG、结构化输出，ngram 这类便宜 draft 就能打得很高

反过来，什么场景收益小：

- **batch 大**：GPU 已经被填满了，投机采样加的那点计算挤不进去，甚至变慢
- **纯创作类长尾数据**：draft 准不了

这对我们很关键：**合作方的实际业务场景是大 batch 还是低延迟 batch=1？** 这个问题没对齐之前，benchmark 怎么跑都可能打偏。
