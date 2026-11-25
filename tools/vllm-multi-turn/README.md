# vLLM multi-turn benchmark (vendored)

`benchmark_serving_multi_turn.py`, `bench_dataset.py` and `bench_utils.py` come from
`benchmarks/multi_turn/` of vLLM at tag `v0.31.0`, the vLLM version in the CUDA image. vLLM's
wheel does not ship them. `scripts/benchmark.py` runs them for suite points with
`driver=multi-turn`.

Local changes, marked `alps-vllm-benchmarks`:

- `send_request`: every request sets `ignore_eos` and `temperature: 0`, so each answer has
  exactly the token count of the conversation config, as in the other suites.
- Reasoning counts as output. With a reasoning parser (GLM-5.3 serves with `glm47`), the
  model's thinking streams as `reasoning_content` before the answer, and upstream reads only
  `content`. Here the first token of either kind ends TTFT, every chunk counts for TPOT, and the
  output count is the server's `usage.completion_tokens` (requested with
  `stream_options.include_usage`). Only the answer goes into the next turn's history, as a chat
  client sends it back.
- When a stream carries an `error` chunk, or ends with no token, the client logs the error or
  the first and last 500 bytes it received, and how long it took. Upstream drops both without a trace.

Upstream: <https://github.com/vllm-project/vllm/tree/v0.31.0/benchmarks/multi_turn>
(Apache-2.0).
