---
name: benchmarking
description: Ideas for optimizing LLM inference speed on a GPU cluster - which server facts to check first, and which scenarios to benchmark (random prompts, real-world text, PD). Use when planning benchmark scenarios or choosing a parallelism config.
---

# Optimizing Inference speed


## You first have to check the type of server you are in:
- Check amount of GPU cards per node
- Check amount of NICs per Node
- Check bandwidth speed of NICs
- Check bandwidth speed of Intranode connection (like NVLink)
- Check if the Nodes are of unified memory type (like GH200 that have fast C2C)
- Check Memory speed of GPU RAM and CPU RAM

## General ideas:
- We understand that TP can help in decode as it removes the memory bottleneck of loading the weights

## Creating scenarios:
- Random prompts:
  - Here the idea is we just check for speed (as the expert balancing should be somewhat uniform). We want to understand the TTFT, ITL, TPOT (can also visualize this as tok/s per user) that each parallelism config can get. 
    - For now we understand that in settings like CSCS Alps (https://docs.cscs.ch/) it seems it can be faster to have PP (https://inferact.ai/news/pipeline-parallelism-in-agentic-era) compared to EP as we don't have NVL72 and paying the internode bandwidth in each MoE layer compared to just in each PP is costly, so one needs to understand when the kernels become the bottleneck (like high concurrencies) so that now it is necessary to instead have all the gpus help in the work (like EP with MoE layers).
  - In random prompts PD doesn't make sense so we just want to see decode and prefill separate.
  - You also always need to check you are using the optimized kernels for that parallelism config
- Real world examples:
  - Here now we care about testing PD also, as it should now help because each prompt and expert balancing will vary
- Multi-turn chat:
  - Here we turn prefix caching on, to see how much each parallelism config reuses the earlier turns. vLLM sends each request to the DP engine with the least load (its running and waiting queues, https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html), not to the engine that has the conversation cached, so more engines get fewer cache hits (on GLM-5.3, TP2×DP8 got about 45–53% of the prompt tokens from the cache, against 79–88% on the single-engine TP4×PP4).


## References
- https://vllm.ai/
- https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html
- https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html
- https://github.com/eth-cscs/alps-extended-images
- https://docs.cscs.ch/
