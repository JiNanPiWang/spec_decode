20260421_112451用RTX PRO 6000测试

20260422_104620用5090，命令为
```bash
python3 -m sglang.launch_server     --model-path /root/autodl-tmp/models/GLM-4-32B-0414-awq     --trust-remote-code     --port 6006     --host 0.0.0.0     --quantization awq     --dtype float16     --attention-backend triton     --mem-fraction-static 0.88
```

20260422_113503用5090测的投机采样，命令为
```bash
python3 -m sglang.launch_server --model-path /root/autodl-tmp/models/GLM-4-32B-0414-awq --speculative-draft-model-path /root/autodl-tmp/models/GLM-4.5-0.6B-v3 --speculative-algorithm STANDALONE --speculative-num-steps 5 --speculative-eagle-topk 1 --quantization awq_marlin --speculative-draft-model-quantization unquant --dtype float16 --attention-backend triton --mem-fraction-static 0.86 --trust-remote-code --port 6006 --host 0.0.0.0
```
