# spec_decode

自研投机采样方案，目标：在 GLM 上超过 sglang 原生投机采样，为合作落地做准备。

## 当前阶段

Phase 0 —— 跑通 sglang + 基线测量。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/00_plan.md](docs/00_plan.md) | 总体计划、阶段划分、给老板的交付物 |
| [docs/01_specdec_intro.md](docs/01_specdec_intro.md) | 投机采样入门：原理和直觉（零基础） |
| [docs/02_sglang_analysis.md](docs/02_sglang_analysis.md) | sglang 投机采样源码分析（五种算法） |
| [docs/03_cpu_draft_poc_plan.md](docs/03_cpu_draft_poc_plan.md) | CPU draft POC 实施计划（老板指定方向） |

## 脚本

- `launch_sglang.sh` — 启动 sglang server 的不同变体（baseline / ngram / standalone / eagle）
- `bench_phase0.py` — 单点 benchmark，batch=1 单一配置（Phase 0 用）
- `bench_baseline.py` — 完整的 batch × seq 矩阵 benchmark（Phase 2 再用）

## Phase 0 快速上手

```bash
# 1. 先编辑 launch_sglang.sh 里的 MODEL_PATH
vim launch_sglang.sh

# 2. 启动基线 server（在一个终端里）
./launch_sglang.sh baseline

# 3. 在另一个终端跑单点 bench
python3 bench_phase0.py --label baseline

# 4. 停掉 baseline server，启 ngram
./launch_sglang.sh ngram
python3 bench_phase0.py --label sglang_ngram

# 5. 对比两份结果
ls outputs/phase0/
```

结果保存到 `outputs/phase0/<label>_<timestamp>/summary.json`。

## 环境

- Target：RTX 5090 + GLM-4-9B-int4
- 后续：8/16 卡 A100 集群 + 全量 GLM5
