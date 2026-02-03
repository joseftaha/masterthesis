# Slurm policy-experiment runner

A self-contained Slurm setup for measuring model-residency, switching,
and batching policies on a shared GPU substrate using vLLM 0.16.0 with
two Llama-3 instruct models. It was developed as part of a diploma
thesis on shared-GPU inference policies, but the runner here stands on
its own: nothing in this directory depends on anything outside it, and
the README below is enough to reproduce a full run on any 4×A40-class
node.

## Quick start

```bash
# 0. venv (Python 3.11 with a CUDA 12.4 toolchain on the build host)
python -m venv .venv-real
source .venv-real/bin/activate
pip install --upgrade pip
pip install -e .

# 1. site identity (Slurm account, HF token, NVMe scratch path)
cp cluster.env.template cluster.env
$EDITOR cluster.env
source env.sh                 # prints nothing on success

# 2. submit the full chain
bash submit_all.sh
```

`submit_all.sh` enqueues, in dependency order:

| # | Job | Resources | What it does |
|---|-----|-----------|--------------|
| 1 | `00_setup` | CPU | builds prompt pools and the seven traces |
| 2 | `01_download_models` | CPU | pre-fetches both checkpoints into `HF_HOME` |
| 3 | `02_measure_primitives` | 4×A40 | warm/cold TTFT, warm/cold E2E, both directed switch costs at the reference operating point (512 in / 128 out, concurrency 1) |
| 4 | `03_run_policy` × 5 policies × 4 interactive traces (20 jobs) | 4×A40 each | residency / switching study (Experiment A) |
| 5 | `04_run_batch` × 2 modes × 3 concurrencies × 3 traces (18 jobs) | 4×A40 each | batching / concurrency study (Experiment B) |
| 6 | `05_aggregate` | CPU | per-trace `summaries.csv` |
| 7 | `06_plot` | CPU | renders the figures via `harness/plot.py` |

Roughly 38 GPU-hours of node time end-to-end on this hardware.

To run a single cell:

```bash
sbatch --export=ALL,POLICY=hybrid_break_even,TRACE=interactive_balanced_mix \
       slurm/03_run_policy.sbatch

sbatch --export=ALL,TRACE=offline_batch_c4_primary,CONC=4,MODE=warm \
       slurm/04_run_batch.sbatch
```

## What's pinned

Everything below lives in `config/experiment.json`. The harness reads
that file at startup; nothing here is a runtime default.

### Hardware and runtime

| Parameter | Value | Source |
|-----------|-------|--------|
| Node | 1 node, 4 × NVIDIA A40 48 GB, local NVMe scratch | `hardware.node_class` |
| Slurm partition | `gpu` (override in `cluster.env`) | `hardware.slurm_partition` |
| Node constraint | `a40` | `hardware.node_constraint` |
| Wall clocks (warm / cold / batch / measure) | 06h / 06h / 04h / 01h | `hardware.time_limit_*` |
| vLLM | 0.16.0, V1 engine (`VLLM_USE_V1=1`) | `software.vllm_version` |
| Python / CUDA / dtype | 3.11 / `cuda/12.4` / `bfloat16` | `software.*` |
| Determinism env | `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `env.sh` |

### Models

| | Primary (8B) | Secondary (70B) |
|---|---|---|
| HF repo | `meta-llama/Meta-Llama-3-8B-Instruct` | `casperhansen/llama-3-70b-instruct-awq` (AWQ-INT4 build of `meta-llama/Meta-Llama-3-70B-Instruct`) |
| Revision | `main` | `main` |
| Quantization | bf16 | AWQ-Marlin INT4 (`--quantization awq_marlin`) |
| Tensor parallel | 1 | 4 |
| `gpu_memory_utilization` | 0.85 | 0.90 |
| `max_model_len` | 4096 | 4096 |
| Mean VRAM | ~18 GB | ~50 GB (multi-GPU as required) |
| Chat template | native Llama-3 instruct, applied server-side via `/v1/chat/completions` | same |

### Decoding

| Parameter | Value |
|-----------|-------|
| `temperature` | `0.0` |
| `top_p` | `1.0` |
| Engine seed (`vllm --seed`) | `12345` |
| Per-request seed | `12345 + req_id` (so a given `req_id` is bit-identical across policy runs) |
| `max_tokens`, interactive | 512 |
| `max_tokens`, batch | 128 |

### Primitive measurement

| Parameter | Value |
|-----------|-------|
| Reference input / output tokens | 512 / 128 |
| Concurrency | 1 |
| Warmup requests | 3 (discarded) |
| Measured requests | 10 per cell |
| Cold definition | runtime alive, model loaded from local NVMe; cold TTFT = `load_s + post_load_ttft` |

Reference values used as a sanity check for the measured primitives
(written to `config/experiment.json -> primitives_measurement.expected`):

| | 8B | 70B |
|---|----|-----|
| warm TTFT | 0.55 s | 1.00 s |
| warm E2E  | 2.20 s | 3.90 s |
| cold TTFT (incl. load) | 10.0 s | 16.0 s |
| cold E2E              | 12.1 s | 21.0 s |
| switch 8B → 70B | 10.0 s |  |
| switch 70B → 8B |        | 6.0 s |

### Policies (Experiment A)

| Code | Policy | Init resident | Behaviour |
|------|--------|---------------|-----------|
| P0 | `all_cold` | none | every decision sets `force_reload=true`; controller stops vLLM before each request |
| P1 | `static_hot_8B` | 8B | 8B stays resident; 70B served cold; never switches |
| P2 | `static_hot_70B` | 70B | symmetric control |
| P3 | `reactive_switch_on_miss` | 8B | switches on every miss |
| P4 | `hybrid_break_even` | 8B | switches when `(│Wb│+1)·Δb ≥ Tab + Tba`; bounded-wait safeguard `τmax = cold_e2e(b)` |

P4's threshold and `τmax` are read from the measured `primitives.json`,
not hard-coded.

### Concurrency points (Experiment B)

| Parameter | Value |
|-----------|-------|
| Hot-path model | `Meta-Llama-3-8B-Instruct` |
| Concurrencies | `[1, 4, 8]` |
| Cold baseline | included |
| Submission control | client-side `asyncio.Semaphore(C)` |
| vLLM batching knobs | left at defaults: concurrency is the only varying factor |

## Datasets

Two HuggingFace datasets supply prompts. Both are loaded once on a CPU
node, filtered, sampled into a fixed pool, and written to JSONL. The
JSONL files are what the trace builder and the run-time harness read;
the upstream datasets are not touched again.

### Sources

| Pool | HF repo | Split | Revision | License |
|------|---------|-------|----------|---------|
| `sharegpt_pool.jsonl` | `anon8231489123/ShareGPT_Vicuna_unfiltered` | `train` of `ShareGPT_V3_unfiltered_cleaned_split.json` | `main` | CC-BY-NC-4.0 (research only) |
| `cnn_dailymail_pool.jsonl` | `abisee/cnn_dailymail` | `test` of config `3.0.0` | `3.0.0` | Apache-2.0 / public-domain |

Revision pins live in `datasets/build_prompts.py` (`DATASET_REVISIONS`)
and a copy is written to `datasets/cache/dataset_revisions.json` next
to the generated pools.

### Prompt extraction

ShareGPT (interactive workload). Each conversation's first turn whose
`from` is `human` or `user` is taken and its `value` becomes the
prompt. Conversations starting with the assistant or with empty user
text are skipped.

CNN/DailyMail (batch workload). The `article` field is wrapped in a
fixed template:

```
Summarize the following news article in 2-3 sentences.

Article:
{article}

Summary:
```

The reference `highlights` is tokenised separately and stored as
`ref_output_tokens`; the trace builder uses it to sample realistic
output lengths.

### Tokenisation and length filters

All token counts use the `meta-llama/Meta-Llama-3-8B-Instruct`
tokenizer, which is the same one the served models use. Pool filters:

| Pool | Min input | Max input |
|------|-----------|-----------|
| ShareGPT | 20 | 1500 |
| CNN/DailyMail | 400 | 2600 |

The ShareGPT cap of 1500 is loose on purpose: the workload spec calls
for ~1% of prompts above 900 input tokens, so cutting tighter would
remove the long tail.

### Sampling procedure

For each pool:

1. Iterate the upstream split in source order, keeping rows that pass
   the prompt extraction and length filter.
2. Stable-sort by `(input_tokens, prompt[:64])`. This removes any
   dependency on the upstream HF row order.
3. Shuffle with `random.Random(20240501)`. One isolated RNG per pool;
   `20240501` is the only entropy source.
4. Take the first `N` rows. `N = 4000` for both pools.
5. Write JSONL: `{"prompt": str, "input_tokens": int,
   "ref_output_tokens": int}`. `ref_output_tokens` is omitted from the
   ShareGPT pool.

### Pool sizes vs trace usage

| Pool | Pool size | Used by | Picks per full run |
|------|-----------|---------|--------------------|
| `sharegpt_pool.jsonl` | 4000 | 4 interactive traces × 1000 requests | 4000 |
| `cnn_dailymail_pool.jsonl` | 4000 | 3 batch traces × 1000 requests | 3000 |

Picks are uniform with replacement, so a trace will see some
`prompt_idx` values more than once (~110 collisions per 1000 picks at
this pool size). Across traces a given prompt may also recur, which is
deliberate: every policy run sees the same prompt on the same arrival,
and the per-policy comparison is therefore against a fixed wire-level
workload.

If you regenerate the pools, leave the seed, the filter bounds, the
sort key, and the dataset revisions alone unless you also bump the
`run_id` you tag results with. Any of those changes alters the
workload at the byte level.

## Traces

Each trace is built by an isolated `random.Random` whose seed is
`20240501 + sum(ord(c) for c in trace_name)`, set up in
`traces/build_traces.py`. A given trace name with the master seed
above produces the same CSV every time.

Rows carry `(prompt_idx, prompt_pool)` rather than the prompt itself.
The harness loads the pool JSONL once and looks the prompt up by
index, so traces stay valid even if a future harness change rewrites
the request path.

### Inventory (1000 rows each)

| Trace | Family | Composition | Submission shape | Mean IAT (s) | Output tokens |
|-------|--------|-------------|------------------|--------------|---------------|
| `interactive_sparse_tail` | A | 80 % 8B, 20 % 70B | exponential IAT | 1.95 (P10 0.91, P90 3.20) | lognormal at 412 (8B) / 429 (70B), capped at 512 |
| `interactive_bursty_tail` | A | 80 % 8B, 20 % 70B | 8B background runs of 35–80 with 70B bursts of 5–14 | bg ~2.07; in-burst ~0.27 | as above |
| `interactive_popularity_shift` | A | 80/20 (req 0–399), 50/50 (400–599), 30/70 (600–999) | exponential IAT | 1.62 | as above |
| `interactive_balanced_mix` | A | 50 / 50 with run-length 1–3 | exponential IAT | 1.50 | as above |
| `offline_batch_c1_primary` | B | 100 % 8B | single-request waves | mean wave gap 8.47 (P10 6.70, P90 9.36) | jittered around `ref_output_tokens` ~56, capped at 128 |
| `offline_batch_c4_primary` | B | 100 % 8B | 4-request waves | mean gap 8.25 (P10 6.55, P90 9.03) | as above |
| `offline_batch_c8_primary` | B | 100 % 8B | 8-request waves | mean gap 8.21 (P10 6.48, P90 9.43) | as above |

Per-trace 8B / 70B marginals are pinned by a deterministic post-pass
in `traces/build_traces.py:_enforce_marginal`. With seed `20240501`:

| Trace | Target 70B share | Actual |
|-------|------------------|--------|
| `interactive_sparse_tail` | 0.200 | 0.200 |
| `interactive_bursty_tail` | 0.200 | 0.200 |
| `interactive_popularity_shift` | 0.460 | 0.460 |
| `interactive_balanced_mix` | 0.500 | 0.484 (run-length 1–3 means it can't always land on 500/500) |
| `offline_batch_c{1,4,8}_primary` | 0.000 | 0.000 |

## Layout

```
.
├── README.md
├── env.sh                     sources cluster.env, applies defaults
├── cluster.env.template       site identity, copy + edit
├── pyproject.toml             vLLM 0.16.0 + client deps
├── submit_all.sh              dependency-chained orchestrator
├── config/experiment.json     pinned configuration
├── datasets/build_prompts.py  ShareGPT + CNN/DailyMail pool builder
├── traces/build_traces.py     trace generator
├── harness/
│   ├── vllm_proc.py           vLLM lifecycle wrapper
│   ├── client.py              streaming /v1/chat/completions client
│   ├── policies.py            P0..P4
│   ├── controller.py          Experiment A driver
│   ├── run_batch.py           Experiment B driver
│   ├── measure_primitives.py  warm / cold / switch primitives
│   ├── aggregate.py           summaries.csv compiler
│   ├── plot.py                figure renderer
│   └── download_models.py     pre-fetch HF weights
└── slurm/
    ├── 00_setup.sbatch
    ├── 01_download_models.sbatch
    ├── 02_measure_primitives.sbatch
    ├── 03_run_policy.sbatch
    ├── 04_run_batch.sbatch
    ├── 05_aggregate.sbatch
    ├── 06_plot.sbatch
    └── lib/env_snapshot.sh
```

## Output

```
runs_real/
├── primitives.json
├── runs/<trace>/                         Experiment A
│   ├── <policy>__details.csv
│   ├── <policy>__switches.csv
│   └── summaries.csv
├── runs_b/<trace>/                       Experiment B
│   ├── <mode>_c<C>__details.csv          policy = "<mode>_c<C>"
│   ├── <mode>_c<C>__switches.csv         header-only (B has no switches)
│   └── summaries.csv
├── plots/
└── logs/<trace>/<policy>/                vLLM stdout + env_snapshot.txt
```

Per-request CSV columns (Experiment A and B):

```
policy, trace_name, req_id, model, input_tokens, output_tokens,
arrival, decision_time, start, first_token, finish,
observed_ttft, observed_e2e, service_mode,
hot_model_before, hot_model_after,
switch_cost, buffer_wait, resource_wait, cold_lane, note
```

`service_mode` is one of `hot_warm`, `switched_warm`, `cold`. Switches
CSV: `policy, trace_name, at_time, from_model, to_model, cost,
reason`. Summaries CSV: `policy, trace_name, n_requests, avg_ttft,
p50_ttft, p95_ttft, avg_e2e, p50_e2e, p95_e2e, cold_count, warm_count,
switch_count, makespan, throughput_req_s, throughput_tok_s`.

## Reproducibility

Beyond the dataset and trace seeds covered above, the run-time side is
pinned as follows:

* vLLM is launched with `--seed 12345`, `temperature=0`, `top_p=1`.
* Per-request seed is `12345 + req_id`, so a given request id produces
  the same tokens regardless of which policy run is replaying it.
* `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are exported
  in `env.sh` to keep CUDA kernels reproducible across reruns.
* Each GPU sbatch writes `env_snapshot.txt` into its log directory:
  git SHA, vLLM version, `pip freeze` (the relevant subset),
  `nvidia-smi`, dataset revisions, model revisions, and the resolved
  env vars (with `HF_TOKEN` redacted).
* Switch and cold timings are taken with a wall clock inside the
  controller; thermal and scheduling variance is captured in the
  `samples` arrays of `primitives.json`. Re-run
  `02_measure_primitives` to bound it.
* Cold TTFT and cold E2E in `primitives.json` are user-visible
  delays and include the load time (`load_s + post_load_ttft`). The
  components (`cold_ready_s_median`, `cold_post_load_ttft_median`)
  are written separately so the breakdown is auditable.

## Policy controller notes

* `all_cold` (P0): every decision sets `force_reload=true`. The
  controller therefore tears down vLLM before each request, including
  back-to-back same-model arrivals. This is the lower-bound case in
  which no model is ever resident.
* `reactive_switch_on_miss` (P3): 8B is prewarmed at init so the first
  arrival is a real "warm or switch" and not "whichever model arrived
  first ends up resident".
* `hybrid_break_even` (P4): WAIT decisions are non-blocking. The
  arrival is enqueued and the trace loop continues; later arrivals
  re-evaluate the threshold; a 0.5 s watchdog handles the bounded-wait
  safeguard between arrivals.

## Statistical scope

`02_measure_primitives` runs 3 warmup + 10 measured repeats per cell
by default (`config/experiment.json -> primitives_measurement`). Bump
`measured_requests` for tighter intervals; medians and the raw sample
arrays are both written, so any further analysis can be done offline.
