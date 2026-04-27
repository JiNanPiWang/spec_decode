# GLM-4.7-Flash 高并发投机采样测试报告

## 1. 测试目的

老板要求评估在 GLM-4.7-Flash 全量模型上 sglang 投机采样在**高并发**场景的真实表现。
之前在 GLM-4-32B-int4 上的所有 benchmark 都是 batch_size=1，没有参考价值；
本次换到 96GB 大显存机器跑 bf16 全量模型，覆盖 1 / 8 / 32 / 64 四个并发档次。

## 2. 测试环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server (96 GB) |
| Driver / CUDA | 590.44.01 / 13.1 |
| PyTorch | 2.9.1+cu128 |
| sglang | 0.5.10.post1 |
| 模型 | GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`, bf16) |
| 模型路径 | `/root/autodl-tmp/models/ZhipuAI/GLM-4.7-Flash` |
| 模型大小 | 56.37 GB (bf16, 47 层 MoE Lite, 64 routed experts + 1 shared, 4 active) |
| KV cache | bf16, 23.72 GB / 470,372 tokens |
| context_len | 202,752 |

> 注：sm_120 上当前 PyTorch 编译目标是 CUDA 12.8，启动时 sglang 会打 "SM 12.x requires CUDA >= 12.9" 警告，
> 但 cuda graph 捕获和正常推理都不受影响（5090 上之前同样的警告，cuda graph 必须禁用；
> Server 版 RTX PRO 6000 上 cuda graph 可正常启用）。

### sglang 关键配置

| 项 | baseline | nextn |
|---|---|---|
| `speculative_algorithm` | 无 | `NEXTN` (sglang 内部映射为 `EAGLE`) |
| `speculative_draft_model_path` | – | 同 target 路径（用模型自带的 MTP 层做 draft） |
| `speculative_num_steps` | – | 3 |
| `speculative_eagle_topk` | – | 1 |
| `speculative_num_draft_tokens` | – | 4 |
| `mem_fraction_static` | 0.85 | 0.85 |
| `cuda_graph` | 启用 (bs 1…256) | 启用 (bs 1…256, 含 draft cuda graph) |
| `piecewise_cuda_graph` | 启用 | sglang 默认禁用 (spec v1 模式) |
| `max_running_requests` | 2048 (默认上限) | **48** (sglang 在 spec 模式下强制下调) |
| `disable_overlap_schedule` | False | **True** (spec v1 不支持 overlap scheduler) |

## 3. 测试方法

用 sglang 自带的 **`sglang.bench_serving`** 作为压测客户端，
在同一组真实 chat prompts 上分别压两个 server，取相同并发档位的结果对比。

### 3.1 数据集（`data/prompts_realistic.jsonl`）

ShareGPT 兼容 jsonl 格式，**66 条原始多样化中英 chat prompts × 重复 10 次 = 660 条独立请求**。
每条 prompt 都加了 `[请求编号 NNNN]` 唯一前缀，**避免 sglang prefix-cache 命中导致 throughput 失真**。

prompt 覆盖：编程补全、代码 review、系统/算法概念解释、调试问题、写作翻译、
数学/物理推理、长文总结、架构设计、JSON/YAML 结构化输出，等。

> 一开始用 sglang 自带的 `random-ids` 数据集（纯随机 token id），结果 NEXTN 在 bs=1 上
> accept_len 仅 ~0.49，throughput 比 baseline 低一半 —— 这种 random 序列对 spec decoding
> **不公平**（draft 没有任何 pattern 可学）。换成真实 chat 数据后 accept_len 稳定在 ~1.92，
> 才反映合理的 spec 行为。

### 3.2 并发档位与请求总数

每个并发档位下，请求总数 = `max(64, concurrency × 8)`，确保稳态测量：

| 并发 | num_prompts |
|---|---|
| 1  | 64  |
| 8  | 64  |
| 32 | 256 |
| 64 | 512 |

固定参数：
- 输出长度 `--sharegpt-output-len 256`（强制每条响应 256 token）
- `--apply-chat-template`（走模型自带 chat template）
- `--warmup-requests 2`

### 3.3 复现命令

服务器端，仓库 `/root/autodl-tmp/spec_decode/`：

```bash
# 启动 baseline server
bash launch_glm47.sh baseline

# 启动 NEXTN spec server
bash launch_glm47.sh nextn

# 跑高并发 bench（任一 server ready 后）
OUT_LEN=256 CONC_LIST='1 8 32 64' bash bench_concurrency.sh <baseline|nextn>
```

结果落到 `outputs/glm47_concurrency/<tag>/c{N}.json` + `summary.tsv`。

## 4. 测试结果

### 4.1 throughput / latency 对比

`output_throughput` 是 server 整段 benchmark 的总输出 token / 总时长，单位 token/s。
TPOT = Time Per Output Token (decode 期 inter-token 延迟，p50)。
TTFT = Time To First Token (含 prefill + 排队，p50)。

| 并发 | 指标 | baseline | NEXTN | 比值 |
|---:|---|---:|---:|---:|
| 1  | output throughput (tok/s) | **109.4** | 90.2  | 0.82× |
| 1  | TPOT p50 (ms)             | 9.0       | 10.9  | +21% |
| 1  | TTFT p50 (ms)             | 44.5      | 73.5  | +65% |
| 8  | output throughput (tok/s) | **385.2** | 336.2 | 0.87× |
| 8  | TPOT p50 (ms)             | 20.4      | 22.8  | +12% |
| 8  | TTFT p50 (ms)             | 44.1      | 110.0 | +149% |
| 32 | output throughput (tok/s) | **937.7** | 666.3 | 0.71× |
| 32 | TPOT p50 (ms)             | 33.6      | 46.8  | +39% |
| 32 | TTFT p50 (ms)             | 186.5     | 139.4 | -25% |
| 64 | output throughput (tok/s) | **1598.8**| 873.7 | 0.55× |
| 64 | TPOT p50 (ms)             | 39.4      | 53.9  | +37% |
| 64 | TTFT p50 (ms)             | 145.5     | 3167.6| +2076% |

### 4.2 NEXTN 的 acceptance 数据

来自 server 日志中 640 个 decode batch 的统计：

| 指标 | 数值 |
|---|---|
| 平均 accept_len | **1.92** / 4 (= 48% acceptance rate) |
| accept_len 主要分布 | 1.75–2.20 |
| server 端平均 gen throughput (跨所有 batch) | NEXTN 184.6 tok/s vs baseline 342.5 tok/s |

按理论估计：accept_len=1.92 意味着每个 verify step 平均吐出 (1.92 + 1) ≈ 2.92 个 token，
理论加速 ~2.9×。但 NEXTN 整体 gen tps 只有 baseline 的 **54%**。

## 5. 分析

**现象：sglang 自带 NEXTN 投机采样在 GLM-4.7-Flash 上各并发档次都比 baseline 慢**，
并发越高、相对劣势越大（c=64 throughput 只有 baseline 的 55%）。

主要原因（按贡献排序）：

1. **`max_running_requests` 被强制下调到 48**（baseline 是 2048），
   c=64 时 sglang 必须排队，TTFT p50 从 145 ms 飙到 3.2 s。
   这是 sglang spec v1 的硬性限制，不是参数问题。

2. **每次 verify forward 的成本远超一次 baseline forward**：
   spec mode 下要跑 `num_draft_tokens=4` 的 verify（输入 token 数是 baseline 的 4 倍），
   外加 draft model 的 3 次 forward（`num_steps=3`）。
   accept_len=1.92 意味着每个 verify step 真正吐出的有效 token 不到 3 个，
   边际收益打不平边际成本。

3. **GLM-4.7-Flash MoE Lite 单 forward 已经很快**：
   2048 hidden / 47 层 / 4 active experts 的小活跃量，单步 decode 已经接近 throughput 极限，
   spec 减少 forward 次数的红利在这种小模型上本来就不大。

4. **spec v1 不支持 overlap scheduler**：baseline 默认开启 overlap，spec 必须关。
   仅这一项在高并发下就有可观的 throughput 损失。

5. **MoE Lite 的 piecewise cuda graph 在 spec 模式下默认禁用**，
   而 baseline 启用了 piecewise cuda graph（变长 token 大小都被 graph 化），
   prefill / 长 input 处理上 baseline 有额外优势。

6. **真实 chat prompts 的 draft acceptance 偏低**（~48%）：
   作为对比，sglang 的 NGRAM 在模板化 prompt 上能跑到 5x，但在自由 chat 上 acceptance 同样会暴跌。
   draft 是模型自带的 1 个 nextn layer，本身预测能力有限，且没有针对 chat 微调。

## 6. 结论

* **现状**：sglang 0.5.10 自带的 NEXTN/MTP 投机采样，**配 GLM-4.7-Flash 默认参数（num_steps=3 / num_draft_tokens=4），在 RTX PRO 6000 Blackwell 96GB 上跑高并发场景，比 baseline 慢 13%–45%，并发越高劣势越大**。
* 老板想看的"高并发投机采样性能"，**当前的 sglang 默认 spec 路径并不能带来收益**。
* baseline 在 c=64 已经能跑 **1599 tok/s**（per-req TPOT ~39 ms），是这个机器的当前 ceiling。

## 7. 后续建议

> 这部分是给老板和接下来工作的输入，不在本次测试范围内。

1. **调 NEXTN 的 spec 参数**：试 `num_steps=2 num_draft_tokens=2`（更短 draft，verify 更轻），
   或 `num_steps=5 num_draft_tokens=8`（更激进，但需要更高 acceptance 才赚）。
   最值得做的是用 server log 抓不同参数下的 accept_len × verify_time 平面图，找最优点。
2. **看 sglang 的 spec v2 / overlap scheduler**：log 里提示 `SGLANG_ENABLE_SPEC_V2=True`
   有实验性的 overlap scheduler，能不能恢复一部分高并发吞吐值得评估。
3. **对照实验**：用 NGRAM 在同样数据集跑一遍，看在真实 chat prompts 上 NGRAM 是否同样退化
   （上次在模板化 prompt 上 NGRAM 跑出 5x 是数据集失真，本次 660 条真实 prompts 是公平的对照）。
4. **回到老板的主线**：CPU draft + GPU verify 方案的真正 baseline，应该是
   "高并发下 sglang 内置 spec 算法的最佳吞吐"，本次测试给出的 baseline 数字（1599 tok/s @ c=64）
   就是后续 CPU draft 方案要超越的目标。CPU draft 在 bs=1 上的优势能不能 scale 到高并发，
   这是下一个 milestone 要回答的问题。

## 附录：脚本与产物

仓库内（`/root/autodl-tmp/spec_decode/`）：

- `launch_glm47.sh` — 启动 baseline / nextn / ngram 三种 server，自动 kill 旧进程 + 等待 ready
- `bench_concurrency.sh` — 在指定 server 上扫并发列表，调用 sglang.bench_serving，输出 jsonl + summary.tsv
- `data/prompts_realistic.jsonl` — 660 条真实 chat prompts (66 × 10 + 唯一编号)
- `outputs/glm47_concurrency/baseline/` — baseline 完整结果（c1/c8/c32/c64.json + summary.tsv）
- `outputs/glm47_concurrency/nextn/` — NEXTN 完整结果
- `outputs/glm47_concurrency/baseline_random_ids/` — 早期 random-ids 数据集的 baseline 结果（不参与对比，仅留作参考）
- `outputs/glm47_concurrency/REPORT.md` — 本报告

server 端日志：
- `/root/autodl-tmp/logs/server_baseline.log`
- `/root/autodl-tmp/logs/server_nextn.log`
- `/root/autodl-tmp/logs/bench_baseline.log`
- `/root/autodl-tmp/logs/bench_nextn.log`
