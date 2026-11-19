# GLM-5.3 inference on Alps

Reproducible vLLM serving and benchmarking for `zai-org/GLM-5.3` on Alps:

- NVIDIA GH200 over Slingshot/CXI;
- AMD MI300A over Slingshot/CXI.

Container images are selected by the accelerator platform profiles. CUDA jobs can instead use an
installed uenv as their software stack; the serving scenarios and vLLM configuration are unchanged.
Optional UCCL-EP, Mooncake, and offload components are enabled only by explicit scenarios.

## Software stacks

| Platform | Default stack | Model loader |
|---|---|---|
| GH200 | OCI image from [`config/platforms/cuda.env`](config/platforms/cuda.env) | Run:ai Model Streamer |
| MI300A | OCI image from [`config/platforms/rocm.env`](config/platforms/rocm.env) | Standard vLLM loader |
| GH200 uenv | Explicit `--uenv` reference and `--view` | Platform-profile loader |

For containers, job startup resolves `IMAGE_REF` through the registry and records its immutable OCI
digest. For uenvs, startup resolves the supplied reference through the local uenv repository and
records its immutable SquashFS digest. Resolution failures stop the job before serving.

The EDF remains the authoritative source for runtime environment variables. The launcher exports
its `[env]` table to every `srun` step. Container steps additionally receive the EDF through
`--environment`; uenv steps receive `--uenv` and `--view` instead. Model files use the EDF bind
mount in containers and their shared host path in uenvs.

### UCCL-EP overlay

GH200 container jobs use a newer UCCL-EP than the image provides. The image keeps vLLM, its
patches, and Mooncake (used for disaggregated prefill/decode and the KV store). Only the `uccl`
and `deep_ep` Python packages are replaced.

`config/platforms/cuda.env` pins the upstream commit in `UCCL_OVERLAY_COMMIT`. The overlay is
built by hand, once per commit and image, from the repository root:

```bash
./tools/build-uccl-overlay.sh
```

The build runs on one `normal` GH200 node inside the pinned image and writes a read-only
directory `uccl-<commit>-<image digest>` (first 12 characters of each) under `UCCL_OVERLAY_ROOT`
(Ritom scratch by default), with an `overlay.json` that records the commit, image, build flags,
and any patch from `tools/uccl-overlay/patches/`. The command does nothing when the directory
already exists. Rebuild by hand after changing `IMAGE_REF` or `UCCL_OVERLAY_COMMIT`: a job whose
overlay directory does not exist stops at start and points to this command, instead of loading
an overlay built for another image. After changing the build flags in
`tools/uccl-overlay/build.sh` or adding a UCCL patch, remove the old directory first; the
command prints how.

At job start, the launcher mounts the overlay read-only and puts it first on `PYTHONPATH` for the
vLLM processes. Each node's runtime record stores the overlay's commit, image digest, build flags,
and patches, and the record is invalid if `uccl.ep` or `deep_ep` load from another location.
Benchmark experiment fingerprints therefore separate overlay runs from image-only runs.

Set `UCCL_OVERLAY_COMMIT=` (empty) to use the image's UCCL. uenv jobs do not use the overlay,
because it is built against the container's PyTorch.

### Runtime patches

Fixes that upstream has merged but the image does not have yet are applied as git patches when
a job starts, without rebuilding the image. `RUNTIME_PATCHES` in `config/platforms/cuda.env` lists
the patch files under `patches/`, separated by spaces and applied in order:

| Patch | What it fixes | In upstream from |
|---|---|---|
| `vllm/0001-pr60735-deepep-ht-dbo-record-stream.patch` | DBO with DeepEP high-throughput: an illegal memory access in `ep_scatter` (seen as a hang of all ranks with 2048-token steps) | vLLM 0.33 ([#60735](https://github.com/vllm-project/vllm/pull/60735)) |

The directory of a patch names the package it targets, and the patch uses that project's own
repository paths, so `git format-patch` output from upstream works unchanged. To add one:

```bash
gh api repos/vllm-project/vllm/commits/<sha> -H "Accept: application/vnd.github.patch" \
  > patches/vllm/0002-<short-name>.patch
```

On every node, `scripts/serve-worker.sh` runs `scripts/runtime-patches.py` inside the container
before vLLM starts. The container layer is writable and belongs to the job alone, so the image
never changes. Each patch is applied when it applies cleanly and skipped when the image already
contains it, for example after an image update to a release with the fix. Otherwise the node
stops with git's reason before vLLM starts. Files of the patch outside the package, such as
tests, are ignored. `logs/run-<job>/patches-<role>-node<N>.json` lists each patch with its hash,
its result (`applied` or `already-present`), and the hashes of the patched files. The runtime
record and the benchmark provenance carry the same list, so results show which code they ran.

Set `RUNTIME_PATCHES=` (empty) to run the image unpatched. uenv jobs skip the patches, because a
uenv's packages are read-only; the status file records the skip.

## Repository layout

| Path | Purpose |
|---|---|
| `config/platforms/` | Image, EDF, loader, NUMA, and runtime defaults for each accelerator platform |
| `config/scenarios/` | Declarative, composable serving scenarios |
| `edf/` | Container definitions and authoritative per-platform runtime environment |
| `scripts/submit-serve.py` | Scenario- and software-stack-aware Slurm submission entrypoint |
| `sbatch/serve.sbatch` | Multi-node serving and benchmark workload |
| `sbatch/download.sbatch` | Resumable model download job |
| `benchmarks/` | Standard short-context and production-parity long-context suites |
| `results/` | Versionable benchmark records |
| `logs/` | Ignored Slurm, server, smoke, and cache logs |
| `tools/install-bencher.sh` | Pinned repository-local Bencher CLI installer |
| `tools/build-uccl-overlay.sh` | Builds the pinned UCCL-EP runtime overlay for the CUDA image |
| `patches/` | Git patches applied to the image's packages at job start (`scripts/runtime-patches.py`) |

## Prerequisites and model

Install `uv` once. Host-side Python uses a managed Python 3.12 because the compute-node system
Python is too old for the scripts; no virtual environment is required.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the Hugging Face CLI once:

```bash
uv tool install 'huggingface_hub[cli]'
```

Run submission commands from the repository root. The wrapper always passes
`--uenv-passthrough=ignore`: an active submitter uenv is never inherited, because serving uses the
repository EDF.

The native FP8 checkpoint is pinned to revision
`aca966e4e02791568aa6a4ced368624b3d897f42` and is approximately 704 GiB. Download it to the
default ignored directory, `models/GLM-5.3`:

```bash
sbatch sbatch/download.sbatch
```

Download progress is written to `logs/download-<jobid>.out` and failures to
`logs/download-<jobid>.err`. The download is resumable. Set `MODEL_REVISION` deliberately when
testing another revision.

Serving defaults to `models/GLM-5.3`. A different checkpoint must be visible from every allocated
node:

```bash
./scripts/submit-serve.py --model /path/to/checkpoint
```

The EDF mounts the resolved host directory read-only at `/models/GLM-5.3`. `MODEL_CONTAINER` may
override that absolute container path when an image requires a different location.

## Scenarios

`scripts/scenario.py` resolves layers in this order: `config/scenarios/baseline.json`, the selected
`config/platforms/<platform>.json`, then scenario components from left to right. Platform defaults
own the common CUDA and ROCm NUMA maps, CUDA's 0.88 memory budget, and
`VLLM_ROCM_USE_AITER=1`. A scenario component can still override those defaults; its selected
`platform_overrides` layer is applied immediately after that component's general settings. The
resolver then validates platform and topology constraints and emits the exact vLLM YAML plus
`scenario.env`.

| Scenario | Change | Constraints or notes |
|---|---|---|
| `baseline` (default) | `allgather_reducescatter` | TP=2; CUDA GPU memory 0.88; BF16 KV cache; external KV and prefix caching off; async scheduling on |
| `uccl-ll` | `deepep_low_latency` | UCCL-EP over CXI; CUDA pins `deep_gemm`, GPU memory 0.90; 256-token step budget (small LL buffers) |
| `uccl-ll-tp4` | `uccl-ll` with TP4 x DP4 (EP16) and 64 sequences per engine | Four engines hold the decode suite's 256 requests; control for `uccl-ll-tp4-dbo` |
| `uccl-ll-tp4-dbo` | `uccl-ll-tp4` with dual batch overlap on decode | Splits decode steps above 16 tokens (c128 and up) into two micro-batches so one's LL all-to-all overlaps the other's compute |
| `uccl-ht` | `deepep_high_throughput` | UCCL-EP over CXI; CUDA pins `deep_gemm`, while ROCm retains its platform backend selection |
| `uccl-ht-4096` | `uccl-ht` with a 4096-token step | Fewer prefill steps per long prompt; each HT step pays a large fixed cost |
| `uccl-ht-4096-dbo` | `uccl-ht-4096` with dual batch overlap | Splits each step into two micro-batches so one's HT all-to-all overlaps the other's compute |
| `uccl-ht-8192` | `uccl-ht` with an 8192-token step | An 8K prompt is one prefill step instead of four |
| `uccl-ht-8192-dbo` | `uccl-ht-8192` with dual batch overlap | As `uccl-ht-4096-dbo`, with twice the tokens per micro-batch |
| `uccl-ht-tp4-pp2` | `uccl-ht` with TP4 x PP2 x DP2 (EP8) and an 8192-token step | Each replica spans two nodes, one pipeline stage per node, so a long prompt's chunks overlap across stages as in TP4 x PP4, while the MoE all-to-all spans both replicas (EP8, HT); compare with `uccl-ht-8192` |
| `moe-deep-gemm` | `deep_gemm` expert kernels | CUDA-only; composable with baseline or low-latency all-to-all |
| `mooncake-cpu` | `MooncakeStoreConnector` | Embedded 8 GB CPU segment per rank and prefix caching |
| `eplb` | Expert load balancing | 32 redundant experts with the Gloo communicator |
| `offload-uva` | UVA weight offload | 16 GiB per GPU |
| `offload-prefetch` | Async layer-group offload | Two of every eight layers offloaded; one layer prefetched |
| `speculative-mtp` | MTP speculative decoding | Three speculative tokens |
| `prefix-caching` | Local GPU prefix cache | No external KV connector |
| `dual-batch-overlap` | Dual batch overlap | Overlaps communication and computation |
| `disaggregated-pd` | Mooncake prefill/decode split | Four prefill (HT) plus four decode (LL) nodes, both `deep_gemm`; CUDA only; not composable |
| `disaggregated-pd-prefill-no-async` | `disaggregated-pd` with async scheduling off on prefill | With async scheduling, the prefill engine marks KV ready to send one (eager HT) step late; decode keeps async scheduling |
| `disaggregated-pd-prefill-dbo` | `disaggregated-pd` with dual batch overlap on prefill | Splits prefill steps above 512 tokens into two micro-batches, so one's HT all-to-all overlaps the other's compute; decode (at most 16 tokens per step) unchanged |
| `disaggregated-pd-prefill-4096` | `disaggregated-pd` with a 4096-token prefill step and 32 decode sequences | Doubles the prefill `max-num-batched-tokens` from 2048, and decode `max-num-seqs` (and its equal batch budget) from 16 to 32 per engine |
| `disaggregated-pd-decode-tp4-pp2` | `disaggregated-pd` with decode TP4 x PP2 x DP2 (EP8) | Each decode replica spans two nodes, one pipeline stage per node; the role uses `--nnodes`/`--node-rank` with the mp backend; prefill unchanged |
| `production` | External production vLLM parity | Exactly four CUDA nodes, TP=4/DCP=4, 1M context, FP8 KV cache, 32 sequences, 4096-token scheduler cap, and one MTP token |
| `million-context` | Experimental vLLM 0.30 capacity reproducer | Exactly four CUDA nodes, TP=4, FP8 KV cache, native CPU KV offload; currently fails startup without DCP |

`production` tracks the active vLLM parameters in `glm-5-3-fp8_tp_dp.yaml`: FlashMLA sparse
attention, DCP=4 with `ag_rs`, custom all-reduce enabled, MTP=1, and EPLB and Mooncake disabled.
The benchmark image, Slurm runtime, network environment, and resource settings remain owned by
this repository's platform configuration rather than the Kubernetes deployment.

The resolver rejects the incompatibilities it explicitly encodes; ordered overlays otherwise use
normal last-component-wins deep-merge semantics. On ROCm, `uccl-ll` uses 0.70 GPU memory
utilization and eager execution. On CUDA, `uccl-ht` pins DeepGEMM because automatic selection
chose FlashInfer CUTLASS and stalled distributed engine initialization; the working path used
DeepGEMM with `DeepEPHTPrepareAndFinalize`. DeepEP HT and LL both need DeepGEMM on CUDA, so
`uccl-ht`, `uccl-ll`, and both `disaggregated-pd` roles pin it. ROCm retains its platform-native
MoE selection. Use `moe-deep-gemm` to pin DeepGEMM in other CUDA runs.

DeepEP low-latency sizes its RDMA buffers from `max-num-batched-tokens`: for GLM-5.3 about
12.6 MB per token per GPU, so a 1024-token budget took about 12 GiB from the KV cache (job
5037516 ran out of memory at 2048). A decode step carries one token per running sequence, so the
PD decode roles, which never prefill, set the budget equal to their sequence cap (16 or 32). The
single-mode LL scenarios (`uccl-ll`, `uccl-ll-tp4`) use 256 tokens: their engines also prefill,
and with every slot decoding a budget equal to the slots would leave almost no tokens for a new
prompt. 256 keeps room for prefill, keeps the buffer at about 3 GiB, and lets every slot fit in the
KV cache. The resolver (`size_low_latency_steps` in `scripts/scenario.py`) raises the budget to
`max-num-seqs x (1 + speculative tokens)` when a later layer adds sequences or MTP tokens, as
vLLM requires, and keeps a larger explicit budget such as the one-million-token profile's 1024.

`uccl-ll+million-context` records the attempted native 1,048,576-token target on four GH200 nodes;
it is a failure reproducer, not a supported serving profile. Job 5037564 measured only
15.86 GiB/GPU available for KV cache without offload, versus 52.68 GiB required, and estimated a
315,776-token ceiling. Job 5037589 tried 42 GiB/GPU of UVA weight offload, reached
405,443,520 KiB (about 387 GiB) step RSS while loading, and was OOM-killed. Job 5037650 accepted
a 40 GiB native CPU KV-offload configuration but applied the same local-KV startup check:
15.85--15.87 GiB available and an
estimated 315,392--315,904-token ceiling. Native KV offload therefore did not extend the maximum
single active request on this vLLM/model path.

The existing `mooncake-cpu` component is complementary, not a demonstrated solution to this
limit. Its `MooncakeStoreConnector` externalizes reusable prefix blocks and currently has
`enable_offload=false`; it does not satisfy the observed startup requirement that one active
`max-model-len` request fit local KV capacity. Do not combine it with the reproducer until active
single-request paging is supported and verified.

Evidence from the predecessor investigation found the high-throughput backend unsuitable for
decode, but `uccl-ht` remains available as a prefill-heavy experiment on either platform.
`disaggregated-pd` uses high throughput for prefill and low latency with 0.90 CUDA memory
utilization for decode. A decode step carries one token per running sequence, so decode sets
`max-num-batched-tokens` equal to `max-num-seqs` (16); DeepEP LL sizes its buffers from that cap.

## Submit serving

Use the wrapper rather than invoking `sbatch/serve.sbatch` directly. It resolves the scenario
before submission and requests the correct allocation.

```bash
./scripts/submit-serve.py
```

Node selection follows the scenario constraints:

- exact-size scenarios request their required node count;
- scalable scenarios default to `min_nodes` and accept `--nodes`;
- `disaggregated-pd` derives its allocation from its prefill and decode role counts.

Examples:

```bash
./scripts/submit-serve.py --scenario uccl-ll
./scripts/submit-serve.py --scenario uccl-ll+eplb --nodes 8
./scripts/submit-serve.py --scenario uccl-ll+million-context --smoke-only --time 04:00:00
./scripts/submit-serve.py --scenario production
./scripts/submit-serve.py --scenario disaggregated-pd
./scripts/submit-serve.py --platform rocm
```

To use the installed vLLM 0.31 uenv on GH200 instead of the CUDA container:

```bash
./scripts/submit-serve.py \
  --scenario baseline \
  --platform cuda \
  --uenv 'vllm/0.31:rc1@daint%gh200' \
  --view vllm \
  --smoke-only
```

`--view` defaults to `vllm` when `--uenv` is supplied. The wrapper continues to pass
`--uenv-passthrough=ignore` to `sbatch`, so an unrelated uenv active in the submitting shell cannot
leak into the allocation.

Common Slurm overrides are explicit wrapper options:

```bash
./scripts/submit-serve.py --platform rocm --nodes 8 \
  --account csstaff --partition <partition> --time 02:00:00
```

The default CUDA allocation is four nodes with four GPUs each:

```text
TP = 2
DP = 8
EP = DP x TP = 16
```

Distributed serving steps use `--distribution=block:cyclic` and leave Slurm CPU binding disabled;
vLLM binds accelerator workers to their local NUMA resources. Container runs use
`--environment=<edf>`; uenv runs use `--uenv=<reference> --view=<view>`.
The default single service uses all allocated nodes; its OpenAI-compatible endpoint is served on
port 8000 of the first allocated node. After readiness, the job performs one real chat completion.
Without a benchmark suite or `--smoke-only`, the server remains active until the job is cancelled.

Logs are split by purpose:

```text
logs/serve-<jobid>.out       Slurm standard output
logs/serve-<jobid>.err       Slurm standard error
logs/run-<jobid>/            Scenario, per-node runtime, server, smoke, and cache records
```

Scenario materialization writes one complete vLLM YAML per role-local node under
`logs/run-<jobid>/scenario/`, for example `vllm-single-node0.yaml` or
`vllm-decode-node3.yaml`. Each file includes the model, topology, coordinator, rank, and API
settings used by that process. Workers launch only `vllm serve --config <node-yaml>`; no vLLM
settings are appended by the Slurm launcher.

## Run the standard benchmark

Pass the suite to the submission wrapper. The job exits after readiness, smoke completion, cache
publication, and all benchmark points complete.

```bash
./scripts/submit-serve.py --benchmark-suite benchmarks/standard.tsv
./scripts/submit-serve.py --platform rocm \
  --benchmark-suite benchmarks/standard.tsv
```

The standard suite contains 11 points:

- concurrency scaling at 1024 input and 128 output tokens;
- prefill scaling at concurrency 8 with 1K, 4K, and 16K input tokens;
- decode scaling at concurrency 8 with 128, 512, and 1024 output tokens.

The long-context suite preserves the standard suite for historical comparison and separately tests
32K, 128K, and 256K input contexts with 128 output tokens at concurrency 8. Run it against the
four-node TP=4/DCP=4 production-parity scenario, whose 1,048,576-token context limit,
32-sequence scheduler, and 4096-token batch cap match the external deployment's active vLLM
parameters:

```bash
./scripts/submit-serve.py --scenario production \
  --benchmark-suite benchmarks/long-context.tsv --time 08:00:00
```

Long-context points retain 128 requests and three repetitions so their p99 values have the same
meaning as the standard suite; the explicit eight-hour allocation accounts for that measurement
volume.

Every point uses deterministic greedy sampling (`temperature=0`) with random prompt generation seeded
to zero. Each measured repetition contains at least 128 requests (256 at concurrency 64), so every
reported p99 is derived from at least 128 request observations.

The benchmark requires the server `/tokenize` endpoint, verifies raw and chat-template tokenization
before measurement, and fails if measured input-token totals differ from the expected
post-template lengths. `summary.tsv` records both requested `input_len` and
`effective_input_len`.

Before measurement, the harness runs every suite point once with the same input/output lengths,
concurrency, seed, and sampling settings but an explicit smaller `warmup_prompts` count. The
standard suite uses `min(prompts, max(16, 2 * concurrency))`: at least two concurrency waves and
at least 16 requests, without paying for statistically meaningful sample counts in an unmeasured
phase. This full shape pass retains every suite shape and at least two scheduler waves; its
diagnostic results are retained under `warmup/`.

Measurement then runs three complete suite passes. `summary.tsv` reports the arithmetic mean and
sample standard deviation of each run-level metric. `bencher.json` publishes that mean with
`lower_value` and `upper_value` set to one sample standard deviation below and above it. The batch
script requests four hours to accommodate the warmup and three separate timed passes.

Around every warmup and measured invocation the harness also reads vLLM's `/metrics` from the
engines (the server, or both roles in PD; the proxy has none) and keeps the difference. The
client sees only the whole TTFT; the server splits it into the time a request waited before it
started running (`vllm:request_queue_time_seconds`) and its prefill
(`vllm:request_prefill_time_seconds`). It also records how long each request spent decoding
(`vllm:request_decode_time_seconds`), from its first output token to its last. The manifest
records mean and estimated p50/p90/p99 of these and of the server-side TTFT for every repetition
and engine role, with prefix-cache counts; `summary.tsv` adds `server_mean_queue_ms`,
`server_p99_queue_ms` and `server_mean_prefill_ms` (the prefill role in PD) and
`server_mean_decode_ms` (the decode role in PD). Percentiles come from vLLM's histogram buckets, so they are approximate.

Slurm output uses UTC `[status ...]` lines for scenario materialization, worker launch, readiness,
smoke, cache publication, and benchmark phases. Readiness reports once per minute. Every warmup,
measurement, or cache-publication subprocess reports a heartbeat every 60 seconds and a completion
line with elapsed time.

Each completed run is written under `results/<run-id>/`:

```text
manifest.json                    Run state, provenance, topology, commands, and fingerprints
summary.tsv                      Human-readable metrics for every point
bencher.json                     Bencher custom-measure payload
submitted-batch-script.sbatch    Exact Slurm spool script executed by the job
raw/<point>/repetition-<n>.json  Original vLLM result for each measured repetition
warmup/<point>.json              Unmeasured full-suite warmup result for each point
```

## Persistent JIT cache

Compiler hot paths stay in container-local `/tmp`. Benchmark jobs run the full-suite warmup, then
capture every node's local artifacts and publish an immutable generation under `.cache/jit` before
measurement starts. Jobs without a benchmark suite publish after smoke instead. A compatible later
job hydrates that generation before vLLM starts. Compatibility includes the image, Python and
package versions, CUDA or ROCm version, and accelerator identity; scenarios are not a compatibility
dimension.

Cache capture is a strict barrier for benchmark jobs: capture errors stop the run before the first
measured request. A publication conflict caused by another job advancing the same namespace keeps
that complete generation and records the conflict, but does not discard the already-warmed
allocation or block measurement.

The wrapper exposes the operational cache modes directly:

- `readwrite` (default): hydrate and publish;
- `read`: hydrate without publishing;
- `off`: use clean node-local caches without reading or publishing.

```bash
./scripts/submit-serve.py --jit-cache-mode off
./scripts/submit-serve.py --jit-cache-mode read
./scripts/submit-serve.py --smoke-only
```

`--smoke-only` exits after smoke and cache publication when no benchmark suite is selected.
FlashInfer's rank-local autotune JSON is excluded because partial hits can desynchronize collective
profiling. Snapshots include FlashInfer, Triton, TorchInductor, Torch extensions, vLLM, and
TensorRT-LLM DeepGEMM artifacts.

Controls:

- `JIT_CACHE_ROOT`: shared root, default `.cache/jit`;
- `JIT_CACHE_LOCK_SECONDS`: publication lock deadline, default 300;
- `JIT_CACHE_BUNDLE_SECONDS`: node-bundle deadline, default 1800;
- `JIT_CACHE_KEEP_GENERATIONS`: retained generation count, default 3;
- `JIT_CACHE_SEED_IMAGE_REFS`: explicit comma-separated source images for a migration.

The current CUDA profile carries one explicit seed from its preceding image. Seed reuse still
requires every non-image compatibility field to match, and reused artifacts are republished under
the current image namespace. Missing, invalid, or conflicting snapshots fall back to clean local
caches without changing inference correctness.

## Benchmark provenance and Bencher

`manifest.json` schema 2 records the repository state, selected runtime backend, software-stack
reference and immutable digest, the exact Slurm spool script, model identity, EDF environment
source, resolved scenario, topology, per-node runtime identities, complete executed server and
benchmark commands, cache status, and result hashes.

Benchmark initialization also creates `reproduction/` in the result directory. Its `vllm/`
subdirectory contains byte-for-byte copies of every role/node YAML, `environment.json` contains
the allowlisted environment captured for each process, and `bundle.json` records their checksums.
The allowlist includes every EDF runtime setting, software-stack identity, uenv selection when
applicable, and scenario-declared variables; unrelated process environment is excluded. These
artifacts are created before warmup and remain available when a later benchmark phase fails.

Two fingerprints separate experiment identity from record integrity:

- `experiment_fingerprint` covers relevant tracked and untracked source files, the executed Slurm
  batch-script checksum, model metadata and shard inventory, software stack, EDF environment
  source, scenario, topology, stable runtime identity, the reproduction bundle's normalized
  identity, and suite definition;
- `record_fingerprint` additionally covers the completed output artifacts, including the archived
  batch script and exact reproduction bundle.

Hostnames, timestamps, concrete node names, coordinator addresses, and run-local cache paths are
excluded from experiment identity so equivalent runs on the same allocation shape retain the same
fingerprint. Their exact measured values remain checksummed in the reproduction bundle and are
therefore covered by record integrity.

Install the pinned Bencher CLI into `tools/bin/<arch>/`:

```bash
tools/install-bencher.sh
```

The installer supports aarch64 and x86_64, verifies the downloaded binary against the checksum in
the script, and does not modify `$HOME` or `PATH`. To upgrade, change `BENCHER_VERSION` and the
corresponding checksums in `tools/install-bencher.sh`.

Upload a completed result; only `BENCHER_API_KEY` is required at publication time:

```bash
BENCHER_API_KEY=bencher_run_... scripts/publish-bencher.sh results/<run-id>/
```

Publication metadata comes from the manifest:

- project: `glm-5-3-inference` by default;
- testbed: `alps-gh200` or `alps-mi300a`;
- benchmark variant: `glm53-<platform>-<nodes>n-<gpus>g-tp<tp>-dp<dp>-ep<ep>-<scenario>`;
- branch and hash: the measured repository state;
- publishable: true only for a clean worktree.

The publisher prefixes every benchmark name with the variant. It does not pass the variant as
Bencher's `--spec`: that option selects remote Runner hardware, requires `--image`, and would
incorrectly rerun rather than upload the measurements produced on Alps.

Known generated paths and root-level Linux core dumps are excluded from source identity. Other
untracked files remain fingerprinted and make the run non-publishable because they may affect the
measured code or configuration.

`BENCHER_PROJECT`, `BENCHER_TESTBED`, `BENCHER_BRANCH`, `BENCHER_SPEC`, and `BENCHER_HASH` may
deliberately override manifest values. The publisher rejects incomplete runs and dirty-tree runs
unless `BENCHER_HASH` is supplied explicitly.

Custom measure names encode their units:

- `request-throughput-req-per-sec`;
- `output-throughput-tok-per-sec`;
- `total-throughput-tok-per-sec`;
- `mean-ttft-ms`, `p99-ttft-ms`;
- `mean-tpot-ms`, `p99-tpot-ms`.

## Measurement boundaries

This repository provides reproducible serving and measurement plumbing, not performance claims.
The four-node baseline floor is currently applied to both platforms. It comes from predecessor
GH200 work; MI300A memory fit and optimal parallelism remain unmeasured. Results belong in
`results/`, and launcher defaults should change only after measurement.
