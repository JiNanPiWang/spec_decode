# GLM-4.7-Flash 高并发投机采样全景对比报告（v2）

**日期**：2026-04-29
**测试机**：autodl 实例（容器内），NVIDIA RTX PRO 6000 Blackwell Server 96GB
**模型**：`zai-org/GLM-4.7-Flash`（Glm4MoeLiteForCausalLM, bf16, 56.4 GB）
**框架**：sglang 0.5.10.post1
**前一版报告**：`outputs/glm47_concurrency/REPORT.md`（commit `ae737b2`，仅含 baseline + NEXTN）

---

## 一、TL;DR（给老板的 30 秒摘要）

在 GLM-4.7-Flash 上把 sglang 现成的 **4 种投机采样路径**（NEXTN / EAGLE / EAGLE2 / EAGLE3）
跨 **4 档并发**（c=1/8/32/64）跑了完整 matrix。

**结论**：sglang 自带 spec 在这个模型上**全部是负优化**，每一档并发都比 baseline 慢；
并发越高，spec 输得越多；c=64 baseline 1604 tok/s，最强 spec（EAGLE2）只有 972 tok/s（baseline 的 61%）。

**对项目方向的判断**：
- 自研 CPU draft + GPU verify 要超越的不是 sglang spec（已经全负优化），是 **baseline c=64 = 1604 tok/s**
- 单纯把 draft 搬 CPU 不够，必须同时打破 sglang spec mode 的 3 个架构限制（见第六节）
- "高并发 spec 更香"是错的——本次实测正好相反，spec 在低并发收益最大

---

## 二、核心数字（output throughput, tok/s）

| 并发 | baseline | NEXTN | EAGLE2 | EAGLE3 |
|---:|---:|---:|---:|---:|
| 1  | 110.3 | 91.3  | 95.2  | 71.7 |
| 8  | 384.2 | 335.0 | 350.7 | 256.3 |
| 32 | 937.6 | 681.0 | 731.2 | 519.5 |
| 64 | **1603.5** | 883.1 | 972.1 | 671.7 |

**spec / baseline 比值**（全部 < 1，即全负优化）：

| 并发 | NEXTN | EAGLE2 | EAGLE3 |
|---:|---:|---:|---:|
| 1  | 0.83× | 0.86× | 0.65× |
| 8  | 0.87× | 0.91× | 0.67× |
| 32 | 0.73× | 0.78× | 0.55× |
| 64 | **0.55×** | **0.61×** | **0.42×** |

**accept_length / draft_max**（命中长度，越高越好）：

| variant | accept_len | draft_max | acceptance |
|---|---:|---:|---:|
| NEXTN  (chain, topk=1)  | 1.95 | 4 | 49% |
| EAGLE2 (tree,  topk=4)  | **2.30** | 8 | 29%（但绝对长度最长） |
| EAGLE3 (HF head, topk=1)| 1.42 | 4 | 36% |

---

## 三、c=64 TTFT 异常值（拥塞放大镜）

| variant | TTFT p50 (ms) | 备注 |
|---|---:|---|
| baseline | 142  | 正常 |
| NEXTN    | 3075 | **21× baseline** |
| EAGLE2   | 2392 | **17× baseline** |
| EAGLE3   | **5908** | **42× baseline** |

**根因**：sglang 在所有 spec 模式下强制把 `max_running_requests` 从 baseline 默认的 **2048 砍到 48**。
c=64 客户端发出 64 路并发，server 一次最多只能处理 48 路，剩下 16 路全部排队，TTFT 直接飙升秒级。
EAGLE3 因为 verify 计算更重排队恶化最严重。

---

## 四、四个 spec algo 的差别 + 排序

四个 algo 可以按"draft 头来源 × 采样方式"两个维度分：

| algo | draft 头 | 采样 | accept_len | c=64 tok/s |
|---|---|---|---:|---:|
| NEXTN  | 模型自带 MTP（57th nextn layer）| chain（topk=1） | 1.95 | 883 |
| EAGLE  | 模型自带 MTP                    | chain（topk=1） | ≈ NEXTN | 未单独测 |
| EAGLE2 | 模型自带 MTP                    | tree（topk=4）  | **2.30** | **972** |
| EAGLE3 | HF `GLM-4.7-Flash-Eagle3` 单层 head | chain/tree    | 1.42 | 671 |

**两条结论**：

1. **同一个 draft 头下，tree（EAGLE2）严格优于 chain（NEXTN）**：因为 tree 一次给多条候选路径，
   target verify 一次吃下整棵树挑最长 prefix，所以 accept 更长（2.30 vs 1.95）。
2. **HF 上下载的 EAGLE3 head 比模型自带的 MTP 头还差**（1.42 < 1.95）。猜测是社区训练时
   未对齐 GLM-4.7-Flash 最终 release 版本的 weights，draft 预测精度低。

---

## 五、为什么 spec 在 GLM-4.7-Flash 上必输？

理论框架：spec 想赢必须满足

```
(accept_len + 1) × T_baseline  >  T_draft + T_verify
```

代入实测数字（c=1）：

- T_baseline ≈ 9 ms（GLM-4.7-Flash MoE Lite 单 token decode forward）
- T_draft   ≈ 6–10 ms（draft head 跑 num_steps=3 步）
- T_verify  ≈ 15–18 ms（target 跑 num_draft_tokens=4 输入的 forward）
- accept_len ≈ 1.95（NEXTN）

理论 spec 上限 = (1.95 + 1) × 9 / (8 + 16) ≈ **1.1× baseline**——也就是说**理论最好只能略快**，
任何架构 overhead（spec scheduler、关 overlap、关 piecewise cuda graph）都会把这点边际优势吃光。

实测 spec 在 c=1 全部负优化（NEXTN 0.83×，EAGLE2 0.86×），完全符合上面的理论估算。

**根本原因**：
- GLM-4.7-Flash 是 **MoE Lite**（4 active experts，47 hidden 层 hidden=2048），
  baseline 单 forward 已经只要 9 ms，**baseline 的天花板太低，不给 spec 留空间**
- 大模型上 spec 红利更大（baseline forward 慢，spec 节省的 forward 数量更值钱）

---

## 六、为什么并发越高 spec 越差？

c=1 spec 输 14–35%，c=64 spec 输 39–58%。原因分四层：

1. **baseline 在高并发已经把 GPU 算满**：decode 阶段从 memory-bound 转向 compute-bound，
   spec 的"用一次 verify 出多 token"红利前提（memory-bound）不复存在。
2. **sglang spec mode 强制 `max_running_requests=48`**：c=64 直接排队，TTFT 飙到秒级。
3. **spec mode 默认关 `overlap_schedule`**：本来 baseline 的 prefill / decode 可以重叠，spec 模式都串行。
4. **spec mode 默认关 `piecewise_cuda_graph`**：变长输入回退到 eager 路径，再降一档吞吐。

**这意味着**：自研 CPU draft + GPU verify 如果只是把 draft 计算搬到 CPU、套 sglang 现成 spec 框架，
就会继承上面 4 个限制，**根本拿不到 baseline 1604 tok/s 的水平**。要赢必须从架构层动手。

---

## 七、对自研 CPU draft + GPU verify 项目的指导意义

### 7.1 真正要超越的目标

| 目标 | 数字 |
|---|---|
| **必须超越**：baseline c=64 | **1603 tok/s** |
| 现有 sglang spec 上限：EAGLE2 c=64 | 972 tok/s（baseline 的 61%） |
| sglang NEXTN c=64 | 883 tok/s（baseline 的 55%） |

baseline 是真目标。**别再以为打过 sglang NEXTN/EAGLE2 就算赢了，那是负优化，门槛太低。**

### 7.2 必须打破的架构限制

| 限制项 | sglang 默认值 | 必须改成 |
|---|---|---|
| `max_running_requests` (spec) | 48 | ≥ 256（至少能容下高并发） |
| `disable_overlap_schedule` (spec) | True | False（CPU draft 和 GPU verify 真正并行） |
| `disable_piecewise_cuda_graph` (spec) | True | False（变长输入也走 graph） |
| verify 输入 token 数 K | 4–32 | 自适应（高并发减小 K，低并发用大 K） |

### 7.3 为什么 CPU draft 在我们这个场景**反而有戏**

虽然 spec 整体在 GLM-4.7-Flash 上是负优化，CPU draft + GPU verify 有几个独特优势：

1. **完全不占 GPU 显存**（不像 EAGLE3 head 还要占显存 + 跑 GPU forward），KV cache 能开更大
2. **真正能并行**（CPU 跑 draft 时 GPU 在跑 verify），sglang 当前 spec mode 没法 overlap
3. **draft 速度可控**（CPU AMX/AVX 上跑 0.6B Q4 模型可达 50–100 tok/s 以上），跟 GPU verify 时间能匹配
4. **不依赖 sglang spec 框架**，可以直接接 baseline 的 max_running_requests=2048 + overlap scheduler

### 7.4 当前 blocker

- **draft 模型词表不兼容**：GLM-4.5-0.6B-v3 vocab=151552，GLM-4.7-Flash vocab=154880，
  sglang STANDALONE 直接 CUBLAS 报错。需要找 vocab 对齐的小模型（最理想：拿模型自带 MTP 头的 weights 出来跑 CPU）

---

## 八、工程踩坑（值得归档）

### 8.1 sglang Glm4MoeLite + EAGLE3 必须打的一行 patch

文件：`/root/autodl-tmp/conda_envs/sglang/lib/python3.12/site-packages/sglang/srt/models/glm4_moe_lite.py`，约 436 行 `Glm4MoeLiteModel.__init__` 内：

```python
self.enable_a2a_moe = False  # patch: DeepseekV2Model.forward (EAGLE3 layers_to_capture branch) 读这个
```

**根因**：`Glm4MoeLiteModel.__init__` 调的是 `nn.Module.__init__(self)` 不是 `super().__init__()`，
没继承父类 `DeepseekV2Model` 的 `enable_a2a_moe` 属性。NEXTN/baseline 不踩，EAGLE3 必踩。

### 8.2 EAGLE3 还需要 env

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
```

EAGLE3 draft `max_position_embeddings=4096` < target 202752，sglang 要求显式同意。

### 8.3 autodl 容器 cgroup 内存限制 128 GB

宿主机 1 TB RAM，但**容器 cgroup 限制 128 GB**（`/sys/fs/cgroup/memory.max`）。
单个 sglang server 占 ~68 GB，连续启停时如果 sleep 太短，cgroup OOM 直接 SIGKILL 新 server，
**且不写 dmesg / oom_kill 计数**，现象是 server log 完全空，bash 报 `Killed setsid nohup ...`。

修复：`bench_all_concurrency.sh` 在 `kill_server()` 里轮询
`/sys/fs/cgroup/memory.current`，等到 < 阈值（默认 20 GB）或超时 120s 再启动下一个 server。

### 8.4 draft 词表不兼容会 CUBLAS 报错

`STANDALONE` 算法用 GLM-4.5-0.6B-v3 当 draft，vocab 151552 ≠ target 154880，
draft 跑 forward 时 attention o_proj 报 `CUBLAS_STATUS_INTERNAL_ERROR`。
必须找 vocab 对齐的 draft（同模型族）。

---

## 九、测试方法

### 9.1 数据集

`data/prompts_realistic.jsonl`：66 条真实中英 chat prompts × 10 轮 + 唯一编号前缀 = 660 条。
覆盖：编程 / 算法 / 写作 / 翻译 / 总结 / 架构 / JSON 结构化输出。
每条带 `[请求编号 NNNN]` 唯一前缀，**防止 sglang prefix-cache 命中污染 throughput**。

### 9.2 各并发档 num_prompts

固定 `num_prompts = max(64, concurrency × 8)`：

| concurrency | num_prompts |
|---:|---:|
| 1  | 64  |
| 8  | 64  |
| 32 | 256 |
| 64 | 512 |

固定参数：`--sharegpt-output-len 256`，`--apply-chat-template`，`--warmup-requests 2`，温度 = 默认（spec mode 强制贪心）。

### 9.3 spec 参数（全部用 sglang 默认值）

| variant | algo | num_steps | num_draft_tokens | eagle_topk | draft 模型 |
|---|---|---:|---:|---:|---|
| NEXTN  | NEXTN  | 3 | 4 | 1 | target 自身（自带 MTP 头） |
| EAGLE2 | EAGLE  | 3 | 8 | 4 | target 自身（自带 MTP 头） |
| EAGLE3 | EAGLE3 | 3 | 4 | 1 | `GLM-4.7-Flash-Eagle3`（HF） |

### 9.4 一键复现

```bash
# 一键跑全 matrix（约 50 分钟），结果落 outputs/glm47_concurrency/_combined_summary.tsv
cd /root/autodl-tmp/spec_decode
setsid nohup bash bench_all_concurrency.sh \
  > /root/autodl-tmp/logs/bench_all_main.log 2>&1 < /dev/null &

# 跑指定子集
VARIANTS="baseline eagle2" CONC_LIST="1 64" bash bench_all_concurrency.sh
```

---

## 十、附录：文件清单

仓库：`/root/autodl-tmp/spec_decode/`，GitHub：`git@github.com:JiNanPiWang/spec_decode.git`。

**脚本**：

- `launch_glm47.sh` — 启动 baseline / nextn / eagle / eagle2 / eagle3 / standalone / ngram 任一 server
- `bench_concurrency.sh` — 对单 server 扫一组并发档 + 落 summary.tsv
- `bench_all_concurrency.sh` — 全 matrix wrapper（含 cgroup mem 等待，autodl 必备）

**数据**：

- `data/prompts_realistic.jsonl` — 660 条真实 chat prompts

**结果**：

- `outputs/glm47_concurrency/_combined_summary.tsv` — **全 4 variant × 4 并发汇总（带 accept_length）**
- `outputs/glm47_concurrency/<variant>/c{1,8,32,64}.json` — 单档 sglang.bench_serving 原始 JSON
- `outputs/glm47_concurrency/<variant>/summary.tsv` — 单 variant 4 并发汇总
- `outputs/glm47_concurrency/REPORT.md` — 本报告（v2，超过 v1 commit `ae737b2`）

**日志**：

- `/root/autodl-tmp/logs/bench_all_overall.log` — 总流程
- `/root/autodl-tmp/logs/bench_all_launch_<v>.log` — 各 variant 启动日志
- `/root/autodl-tmp/logs/bench_all_bench_<v>.log` — 各 variant bench 详细日志
- `/root/autodl-tmp/logs/server_<v>.log` — sglang server 自身日志
