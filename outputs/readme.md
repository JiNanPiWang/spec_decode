20260421_112451用RTX PRO 6000测试

20260422_104620用5090，命令为
```bash
python3 -m sglang.launch_server     --model-path /root/autodl-tmp/models/GLM-4-32B-0414-awq     --trust-remote-code     --port 6006     --host 0.0.0.0     --quantization awq     --dtype float16     --attention-backend triton     --mem-fraction-static 0.88
```

20260422_113503用5090测的投机采样，命令为
```bash
python3 -m sglang.launch_server --model-path /root/autodl-tmp/models/GLM-4-32B-0414-awq --speculative-draft-model-path /root/autodl-tmp/models/GLM-4.5-0.6B-v3 --speculative-algorithm STANDALONE --speculative-num-steps 5 --speculative-eagle-topk 1 --quantization awq_marlin --speculative-draft-model-quantization unquant --dtype float16 --attention-backend triton --mem-fraction-static 0.86 --trust-remote-code --port 6006 --host 0.0.0.0
```

20260422_165233现在开始用5090测另外一个int4量化的模型，前面的数据都有点问题，并发到4就显存不够了。下面是关了cuda graph的结果，可以看20260422_171120
```bash
python3 -m sglang.launch_server     --model-path /root/autodl-tmp/models/glm-4-32b-0414-gptq-int4     --trust-remote-code     --port 6006     --host 0.0.0.0     --quantization gptq_marlin          --mem-fraction-static 0.7          --disable-cuda-graph          --disable-piecewise-cuda-graph
```


20260422_171432测STANDALONE投机采样，能快一点，快的不多
```bash
python3 -m sglang.launch_server     --model-path /root/autodl-tmp/models/glm-4-32b-0414-gptq-int4     --trust-remote-code     --port 6006     --host 0.0.0.0     --quantization gptq          --mem-fraction-static 0.7      --disable-piecewise-cuda-graph      --speculative-draft-model-path /root/autodl-tmp/models/GLM-4.5-0.6B-v3          --speculative-algorithm STANDALONE          --speculative-draft-model-quantization unquant           --disable-cuda-graph
```