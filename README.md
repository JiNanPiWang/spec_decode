# spec_decode

自研投机采样方案，目标：在 GLM 上超过 sglang 原生投机采样，为合作落地做准备。

## 当前阶段

Phase 0 跑通完成；Phase 1 CPU draft demo 已出结论（初步不可行，瓶颈在 draft 模型匹配度）。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/00_plan.md](docs/00_plan.md) | 总体计划、阶段划分 |
| [docs/01_specdec_intro.md](docs/01_specdec_intro.md) | 投机采样原理入门（零基础） |
| [docs/02_sglang_analysis.md](docs/02_sglang_analysis.md) | sglang 投机采样源码分析 |
| [docs/03_cpu_draft_poc_plan.md](docs/03_cpu_draft_poc_plan.md) | CPU draft POC 实施计划 |
| [docs/04_cpu_draft_demo_findings.md](docs/04_cpu_draft_demo_findings.md) | **CPU draft demo 首次测量结论（给老板汇报用）** |

## 脚本

- `launch_sglang.sh` — 启动 sglang server 不同变体（baseline / ngram / standalone / eagle）
- `bench_phase0.py` — 单点 benchmark（batch=1），对比各种 spec 方案用
- `cpu_draft_demo.py` — CPU draft + GPU target 可行性 demo，分开测 prompt eval / gen 耗时
- `bench_baseline.py` — 完整 (batch × seq) 矩阵 benchmark（Phase 2 再用）

## Phase 0 / Phase 1 快速上手

```bash
# 1. 启动 sglang（baseline 无投机）
./launch_sglang.sh baseline

# 2. 在另一终端跑单点 bench（拿 baseline 数字）
python3 bench_phase0.py --label baseline

# 3. 跑 CPU draft demo（需要 sglang 是 baseline 模式）
taskset -c 0-31 python3 cpu_draft_demo.py \
  --draft-gguf /root/autodl-tmp/models/GLM-4.5-0.6B-v3-GGUF/GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf \
  --target-model-path /root/autodl-tmp/models/glm-4-32b-0414-gptq-int4 \
  --target-url http://127.0.0.1:6006/v1/completions

# 4. 切 NGRAM 对比
./launch_sglang.sh ngram    # 先 kill 前一个
python3 bench_phase0.py --label sglang_ngram --warmup 5
```

> Xeon 8470Q 是双路 NUMA（0-51 / 52-103），CPU 任务一定要用 `taskset -c 0-31` 绑到
> 单个 NUMA 节点，否则跨 socket 会让性能掉 10x+。

## 环境

- GPU：RTX 5090（32GB，Blackwell sm_120）
- CPU：Xeon Platinum 8470Q（208 线程，AMX_INT8 + AVX-512）
- Target 模型：GLM-4-32B-0414-gptq-int4
- Draft 模型：GLM-4.5-0.6B-v3（HF）或 GLM-4.5-DRAFT-0.6B-Q4_0.gguf（llama.cpp CPU）
