#!/usr/bin/env python3
"""
生成一个真实 chat prompt 的 jsonl 数据集，sglang custom dataset 格式。
覆盖编程、问答、写作、翻译、推理等多个领域，让投机采样在 natural language 上得到公平评估。
"""
import json
import sys

# 多个主题，每个主题数条 prompt，组合后约 80 条独立 prompt
PROMPTS = [
    # ---- 编程 / 代码补全 / 代码解释 ----
    "用 Python 实现一个 LRU 缓存类，支持 O(1) get 和 put 操作。请给出完整代码并解释关键设计。",
    "Complete the following Rust function that checks if a number is prime, and explain your approach:\n\nfn is_prime(n: u64) -> bool {",
    "下面是一段 Go 代码，请指出其中的并发安全问题并给出修复方案：\n\nvar counter int\nfunc add() { counter++ }\nfor i := 0; i < 1000; i++ { go add() }",
    "用 TypeScript 写一个深拷贝函数，要支持嵌套对象、数组、Date、RegExp、Map、Set，并避免循环引用导致的栈溢出。",
    "给我一个 C++ 模板元编程的例子，演示编译期计算 N 的阶乘，并解释 SFINAE 在其中的作用。",
    "解释 React hooks 中 useEffect 的依赖数组工作原理，给出三个常见误用的例子和修正方法。",
    "用 SQL 写一个查询：从 orders 表按月份汇总销售额，并显示同比增长率。orders 表有 order_date, amount 两个字段。",
    "Write a Python decorator that retries a function up to N times with exponential backoff on failure, and supports both sync and async functions.",
    "用 PyTorch 写一个简单的 Transformer encoder 层，包含 multi-head self-attention 和 feed-forward，并解释 residual connection 的作用。",
    "解释一下 CUDA 的 warp divergence 是什么，以及在 kernel 设计时如何尽量避免？",

    # ---- 系统与基础概念 ----
    "请详细解释一下什么是 Transformer 架构里的注意力机制，以及为什么它在自然语言处理上比 RNN 更有效。",
    "给我讲一下 KV cache 在大语言模型推理中的作用，以及为什么 prefill 和 decode 两个阶段的计算特性不一样。",
    "投机采样（speculative decoding）的核心思想是什么？draft model 和 target model 各自做什么？",
    "解释一下 LoRA 这种参数高效微调方法的数学原理，以及它相比全参微调的主要优劣。",
    "MoE（Mixture of Experts）架构里，为什么在推理时 KV cache 占用反而比 dense 模型大？请详细说明。",
    "什么是量化感知训练（QAT）和后训练量化（PTQ），它们的主要区别是什么？实际工程里怎么选择？",
    "Explain the CAP theorem in distributed systems and give a concrete example for each of the three trade-off categories.",
    "Compare and contrast TCP and QUIC. Why is QUIC sometimes preferred for HTTP/3? What are the tradeoffs in latency, security, and middlebox compatibility?",
    "请详细解释一下 GPU 上 tensor core 和 cuda core 的区别，以及它们分别擅长什么类型的运算。",
    "讲讲 RDMA 是什么，它相比传统 TCP/IP 网络在 AI 训练集群中能带来哪些优势？",

    # ---- 调试 / 实践问题 ----
    "我们在训练一个大模型时，发现 loss 突然变成 NaN，通常有哪些可能的原因？应该怎么排查？",
    "我的服务在高并发下偶尔出现 502 错误，怀疑是上游 nginx 配置问题，给我一个系统化的排查思路。",
    "Python 程序的内存占用持续增长但没有明显泄漏，请列出可能的原因，并说明用什么工具去定位。",
    "我用 docker 跑容器时，每隔几小时就会出现 OOM 被 kill 的情况，但容器内进程看起来内存没涨。可能问题在哪？",
    "Postgres 一个查询走了 seq scan 而不是预期中的 index scan，请列出几个常见原因和检查方法。",
    "Kubernetes pod 一直处于 CrashLoopBackOff 状态，给我一个排查清单。",
    "My laptop is overheating whenever I run docker containers. What are the most likely causes, and what should I check first?",

    # ---- 写作 / 翻译 / 创意 ----
    "写一个 200 字左右的短故事：在一个下雨的黄昏，森林里遇见一只会说话的乌龟。要求有对话、有画面感。",
    "把下面这段话翻译成自然流畅的英文，并保持原文的语气：『投机采样的核心思想是用一个小模型快速猜测未来若干个 token，再用大模型一次性验证。』",
    "请把这段英文翻译成中文，并保持技术准确性：『The core idea of speculative decoding is to use a small draft model to predict multiple future tokens, then verify them in parallel with the large target model.』",
    "请写一段产品发布会的开场白，主题是发布一款面向开发者的代码补全 AI 工具，要求语气自信、有感染力，约 200 字。",
    "帮我写一封正式的英文邮件，向一位海外学者 Dr. Smith 介绍我们的研究方向，希望能开展合作，约 250 字。",
    "用文言文风格写一段关于『修身齐家治国平天下』的现代解读，要求语义清晰、有现代意识，约 200 字。",

    # ---- 推理 / 数学 ----
    "鸡兔同笼，共 35 个头，94 只脚，问鸡和兔子各几只？请给出推导过程。",
    "如果一个城市每天有 30% 的人会出门，5% 的出门者会去某家咖啡店，咖啡店每位顾客平均消费 25 元，城市人口 100 万，估算这家咖啡店的日营业额。请说明每一步推理。",
    "证明：对于任意正整数 n，1+2+3+...+n = n(n+1)/2。请给出两种不同的证明方法。",
    "Prove that the square root of 2 is irrational. Use a proof by contradiction and explain each step in detail.",
    "假设你在一个 100 人的房间里，要赌一下『至少有两人生日相同』，给出概率推导（生日悖论）。",
    "如果一辆汽车以 60 km/h 匀速行驶，前方 200 米处有一个红绿灯刚变红 (持续 30 秒)，问车应该减速到什么速度才能在不停下的情况下到达时绿灯刚好亮？请给出推理。",

    # ---- 长文本 / 总结 ----
    "请概述一下大语言模型的发展历程，从 Transformer 论文 (2017) 到 ChatGPT 出现 (2022)，列出几个关键里程碑及其意义。",
    "总结一下 2010 年代深度学习在计算机视觉领域的主要突破：AlexNet、VGG、ResNet、Inception、EfficientNet 等架构的核心贡献。",
    "Summarize the key arguments in the paper 'Attention Is All You Need' (2017), and explain why it had such a large impact on the field.",
    "请简要梳理一下 GPT 系列模型从 GPT-1 到 GPT-4 的发展，每一代的关键改进是什么？",
    "讲讲 MoE（Mixture of Experts）架构在大模型中的发展史，从 Switch Transformer 到 Mixtral 到 DeepSeek-V3，关键创新分别是什么？",

    # ---- 工程 / 架构设计 ----
    "如果让你设计一个支持每秒百万级写入的实时数据库，需要考虑哪些核心问题？请从存储引擎、复制、一致性、缓存几个角度展开。",
    "When designing a distributed database system that must handle millions of concurrent writes while maintaining strong consistency, what are the core trade-offs and how would you address them?",
    "请设计一个基于 LLM 的客服系统架构，要求支持 10 万 QPS，平均延迟 < 2 秒，给出关键组件和数据流。",
    "Imagine you're tasked with building a vector database from scratch that supports 1 billion 768-dim embeddings with sub-100ms ANN search latency. Walk me through your design choices.",
    "我要做一个边缘端的实时人脸识别系统，硬件是 Jetson Orin。请帮我设计端到端 pipeline，包括模型选型、量化、推理后端。",

    # ---- 历史 / 文化 / 通识 ----
    "The historical impact of the Industrial Revolution extends far beyond the immediate technological changes of the 18th and 19th centuries. In what ways did it reshape global politics, economics, and class structure?",
    "请讲一下中国古代四大发明的产生背景，以及它们对世界文明的影响。",
    "为什么文艺复兴会出现在意大利？请从经济、政治、文化几个角度分析。",
    "请简要介绍一下『五四运动』的历史背景、核心诉求和长远影响。",
    "Discuss the role of the Silk Road in connecting Eastern and Western civilizations, both in terms of trade and cultural exchange.",

    # ---- 商业 / 经济 ----
    "请分析一下『双边市场』的核心特征，并以 Uber 和 Airbnb 为例说明它们如何冷启动。",
    "解释一下『网络效应』和『规模经济』的区别，并各举两个互联网产品的例子。",
    "为什么互联网早期更容易出现垄断？请从用户习惯、数据飞轮、固定成本几个角度分析。",
    "如果让你给一家初创公司做 SaaS 定价策略，请给出三种定价模型的优劣分析（按席位、按用量、按价值）。",

    # ---- 物理 / 科普 ----
    "请用通俗语言解释一下量子纠缠是什么，以及为什么爱因斯坦称之为『鬼魅般的超距作用』。",
    "请解释相对论中的『时间膨胀』现象，并举一个 GPS 卫星的实际例子说明这一效应的工程意义。",
    "Explain how black holes form and why nothing, not even light, can escape past the event horizon. Use intuitive language.",
    "讲讲熵增定律是什么，以及为什么它被称为『时间之箭』。",

    # ---- 数据 / JSON / 结构化输出 ----
    "Generate a JSON object describing a user profile with fields name, email, age, preferences (as a nested object with theme and language). Output only valid JSON, no other text.",
    "把以下书目信息整理成一个 JSON 数组，每个元素含 title, author, year, isbn 字段：\n《三体》刘慈欣 2008 9787536692930\n《活着》余华 1993 9787506365437\n《百年孤独》马尔克斯 1967 9787544253994",
    "Generate a YAML config for a Kubernetes Deployment running 3 replicas of a Python web app on port 8080, with readiness/liveness probes and resource limits.",
    "请用 markdown 表格列出 Python 主流 Web 框架 (Django, Flask, FastAPI, Tornado) 在性能、生态、学习曲线、典型场景四个维度的对比。",
]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "prompts_realistic.jsonl"
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    # 每条 prompt 加唯一序号后缀，避免 sglang prefix-cache 命中导致 throughput 失真
    rows = []
    for r in range(repeat):
        for i, p in enumerate(PROMPTS):
            idx = r * len(PROMPTS) + i
            tagged = f"[请求编号 {idx:04d}] {p}"
            rows.append({"conversations": [{"content": tagged}, {"content": "ok"}]})
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写入 {len(rows)} 条 prompts (基础 {len(PROMPTS)} × {repeat} 重复) -> {out_path}")


if __name__ == "__main__":
    main()
