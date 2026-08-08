# DAS: Distribution-Aware Speculative Decoding for RL Rollouts

Implements the acceleration described in "Beat the long tail:
Distribution-Aware Speculative Decoding for RL Training"
([arXiv 2511.13841](https://arxiv.org/abs/2511.13841)) on top of vLLM's
suffix decoding (`method="suffix"`, backed by
[arctic-inference](https://github.com/snowflakedb/ArcticInference)) — with
one architectural addition the paper does not have: a **centralized
trajectory store** that gathers rollouts from *all* data-parallel workers
and redistributes them, so every engine's draft trees see every GRPO
sibling and every prior epoch, regardless of where those rollouts ran.

Speculative decoding is lossless: drafts are verified by the target model,
so outputs (and reward curves) are identical to non-speculative decoding.
DAS only changes wall-clock time.

## Architecture

```
DRIVER  PyInferenceAgentLoopManager.generate_sequences()   [per training step]
   ├─ SuffixDataService.begin_iteration(step)  ─────────┐
   └─ dispatch rollout batch                            ▼
AGENT-LOOP WORKERS                               SuffixDataService
   generate(): problem hash from prompt ids      (named detached Ray actor)
     engine req id = "dasp{hash}-{uuid}"          - versioned delta log
     on completion: batched fire-and-forget ───►  - sliding window (W iters)
     trajectory push                              - 2x shadow copies (recency)
                                                  - length stats + L/M/S classes
ENGINE REPLICAS (vLLMHttpServer actor each)               │
   delta pump  ◄── get_deltas(since_version) ─────────────┘
     └─ collective_rpc("das_apply_deltas")   [all TP ranks, same step boundary]
   GPU WORKERS: vllm.general_plugins entry point patches the stock
     SuffixDecodingProposer in place; per-problem suffix trees + budgets;
     the decode hot path never touches the network
```

Data flow guarantees:

- **Order-safe replication.** The service compacts each delta batch so that
  add/remove pairs inside the batch cancel; replicas converge to
  byte-identical live sets regardless of when they joined (snapshot resync
  covers restarts and log truncation).
- **TP determinism.** Payloads reach all TP ranks via one `collective_rpc`
  at the same step boundary; transient self-inserts derive only from
  rank-identical batch data.
- **Graceful degradation everywhere.** Any failure (missing arctic, vLLM
  API drift, dead service, verl < 0.9) logs once and falls back to stock
  suffix decoding or plain decoding. DAS can degrade; decoding cannot break.

## Enablement

1. **Runtime env** (`integration/verl/examples/runtime-env-das.yaml`):
   `arctic-inference==0.1.1`, pip-installed `py-inference-scheduler` (gives
   the `vllm.general_plugins` entry point pip metadata in every engine
   process), `PYIS_DAS_ENABLED=1`, `DAS_CONFIG_PATH`.
2. **vLLM spec decode** via Hydra passthrough (see
   `run_qwen2_5-32b_math_das.sh`):

   ```
   +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config=
     '{"method":"suffix","num_speculative_tokens":24,
       "suffix_decoding_max_tree_depth":24,
       "suffix_decoding_max_spec_factor":2.0,
       "suffix_decoding_min_token_prob":0.1}'
   ```

   `num_speculative_tokens` must be ≥ the Long budget in the DAS config.
3. **DAS config** (`configs/das_config.yaml`): window, budgets, classifier
   thresholds, memory caps. Every knob documented inline.

Master switch: `PYIS_DAS_ENABLED`. Unset/`0` = every DAS code path is inert
(bit-identical behavior to a build without this feature).

### verl version matrix

| Capability | verl 0.7.x | verl ≥ 0.9 |
|---|---|---|
| Central trajectory gathering + stats | yes | yes |
| Problem-hash request ids | yes | yes |
| Stock suffix decoding (engine-local) | yes* | yes |
| Intra-batch sibling reuse via transient trees | yes* | yes |
| Cross-engine delta delivery (the DAS core) | no — needs `collective_rpc` / `_get_worker_extension_cls` | yes |

\* if the `engine_kwargs.vllm.speculative_config` passthrough works on your
verl build; verified on 0.9.x.

## Design notes (mapping to the paper)

- **Per-problem trees.** The paper's Fig. 6 shows problem-scoped trees beat
  one global tree on acceptance *and* latency. Problem identity here is the
  hash of the first-turn prompt token ids (GRPO siblings share it; it is
  stable across epochs), smuggled to the proposer inside the request id —
  zero verl API changes. Unknown/non-DAS requests fall back to stock
  behavior per request.
- **Sliding window.** `begin_iteration` (called by the manager once per
  training step) advances the window; sequences older than
  `window_iterations` are evicted store- and engine-side.
- **Recency down-weighting.** Arctic trees score by raw frequency ratios,
  so recency is approximated by *multiplicity*: trajectories from the
  freshest `fresh_iterations` carry a shadow copy (2x count) that is
  removed as they age.
- **Length-aware budgets.** Long/Medium/Short classes from the EMA of
  per-problem generation lengths (computed centrally, shipped with deltas);
  running requests upgrade classes as their observed length crosses the
  thresholds; Short skips drafting entirely. The paper's closed-form Eq. 7
  budget is a planned refinement behind the same `BudgetPolicy` interface.
- **Self-inserts are transient.** Engines insert their own sampled tokens
  immediately (same-batch siblings draft from each other without waiting a
  poll cycle) under a reserved seq-id namespace, and drop them when the
  canonical copies arrive with the next iteration's deltas — so replicas
  never diverge for longer than one iteration.

## Verification

- Unit tests: `uv run pytest tests/unit/speculative/`
- GPU-free end-to-end data path (laptop or Ray head, no vLLM/verl needed):
  `python3 -m integration.verl.das_compat_check`
- On-cluster smoke: prepare the dataset
  (`python3 integration/verl/examples/prepare_deepscaler.py`), then run
  `run_qwen2_5-32b_math_das.sh`; check worker logs for
  `DAS: SuffixDecodingProposer patched in place`, engine logs for
  `DAS: delta pump started`, and nonzero `vllm:spec_decode_*` counters.
- A/B: compare per-step rollout time across (no spec) / (stock suffix) /
  (DAS); reward curves must be indistinguishable — speculative decoding is
  lossless, so any reward drift indicates a bug, not a tuning issue. The
  A/B uses DeepScaleR (long competition-math generations) rather than
  GSM8K/MATH: short-generation workloads mostly classify Short/Medium and
  understate DAS's long-tail gains.

## Observability

`SuffixDataService.get_metrics()` (via any Ray client): iteration, log
version, problem count, total tokens, per-replica lag, pushes received,
snapshots served. Engine-side `das_get_tree_stats` (collective_rpc): tree
count, transient count, applied batches, state version — cross-rank version
mismatch means a replica needs snapshot resync and should be reported.

## Known limits (v1)

- `data_parallel_size > 1` inside a single engine: detected and DAS is
  disabled for that engine (verl's collective_rpc reaches one DP shard).
  The standard verl topology (DP = separate replicas) is fully supported.
- SGLang: not yet (no upstream suffix drafter to feed).
- Recency weighting is the 2x-shadow approximation, not a tunable decay.

## Benchmark results (2026-08, GKE H100/H200)

Setup: verl 0.9.0.dev + vLLM 0.11.0 (ngram host), Qwen3-4B, DeepScaleR,
GRPO n=8, batch 64, max_response 4096, T=1.0, TP=1, 16 engines across two
8-GPU nodes, py-inference-scheduler routing in every arm. 10 training
steps per arm; all arms strip rollout logprobs identically. Speculative
decoding is lossless — reward curves were statistically indistinguishable
across every arm throughout.

### No-repeat ladder (each problem seen once)

| Arm | Config | timing_s/gen | Drafted/round | Accepted/round | Acceptance |
|---|---|---|---|---|---|
| A2 | no spec decode | 46.8 | — | — | — |
| B3 | stock ngram, k=4 | 41.2 | ~4 (blind) | ~1 | low |
| C5 | DAS, ungated | 42.1 | 5.94 | 0.85 | 14.2% |

With no problem repetition the central store cannot contribute (trees ship
after their problem's only appearance), so C5 vs B3 isolates drafting
policy: DAS matches the best cheap drafter while drafting a fraction of
the tokens. For contrast, stock ngram misconfigured at k=24 drafted 23.5
tokens/round at 4.6% acceptance and *lost* to no-spec.

### Epoch run (128 problems x 5 epochs, 2 steps/epoch)

| Arm | timing_s/gen avg | Cold (steps 1-2) | Warm (steps 3-10) |
|---|---|---|---|
| A2e | 47.8 (flat all run) | ~47.7 | ~47.8 |
| C5e | **37.8 (-21%)** | 43.3 (-9%) | **36.4 (-24%)** |

The inflection lands exactly at step 3 — the first step where problems
recur and per-problem trees hold prior-epoch trajectories. Acceptance
rose from 14.2% (no-repeat) to 18.7% with accepted-tokens-per-round up
73% (0.85 -> 1.47): warmer trees make drafts longer and better
simultaneously (the paper's Fig. 4 dynamic). A2e's flat curve is the
control proving the drop is decode speedup, not training-induced length
shrinkage. Best warm steps reached -29%; acceptance was still climbing at
epoch 5.

### End-to-end accounting (Amdahl)

Generation was only ~17% of the 229s training step in this configuration
(small model, conservative micro-batches), so the 21% generation win
passes through as only ~+3.3% whole-step throughput (523 -> 541
tok/s/GPU) and near-zero total job time delta. The dilution is a property
of the benchmark's shape, not the mechanism: at the DAS paper's
rollout-dominant shape (70%+ of step), the same 1.3x generation speedup
projects to ~20%+ end-to-end. Deploy where rollout dominates the step;
judge DAS by `timing_s/gen` (or rollout tokens/sec), not whole-step
throughput.

### Routing x DAS (homogeneous H200 ladder, epoch config)

After pinning workers to H200s (heterogeneous fleets make load-blind
routing turn slower GPUs into systematic stragglers): A3 no-spec 39.2
s/step | C5e2 backpressure+DAS **35.3** | C6b prefix-only+DAS 43.5
(degrading across epochs: sticky problem->engine assignment compounds
length skew with no rebalancing valve). Acceptance: 18.5% vs 19.2% —
sibling co-location buys almost nothing because the central store ships
cross-epoch trees to every engine regardless; affinity only accelerates
the same-step transient window. **Verdict: keep the backpressure-dominant
profile; do not trade load balance for affinity.** Also note: H200s
compressed DAS's warm-epoch margin from 24% (mixed fleet) to ~12% —
faster decode rounds, unchanged Python propose overhead; propose-path
batching is the recovery lever, or larger models (round cost grows,
overhead doesn't).

### Paper-scale run (Qwen3-32B, TP=2, 16k response cap)

A4 no-spec 269.5 s/step (reward 0.670) vs C7 DAS **214.1 (-20.5%)**
(reward 0.711; run variance). Margin expanded from ~12% (4B) to ~20.5%
(32B) on identical hardware — costlier decode rounds amortize the fixed
propose-path overhead — despite *lower* acceptance (15.6% vs 18.5%):
at scale, value-per-accepted-token beats acceptance rate. TP=2 delta
delivery ran clean (zero rank resyncs; identical per-pod counters). The
epoch warm-up flattens at 16k: own-context self-repetition and same-step
sibling reuse dominate, so cross-epoch trees add proportionally less.
Caveat: colocated 32B training (micro-batch 1) pushed total step to ~25
min, so rollout share — and thus end-to-end gain — stayed small; the
generation win transfers fully only where rollout dominates the step.

### Deep-epoch run (Qwen3-32B, DeepMath-103K band 4-7, 16k cap, 10 epochs)

A5 no-spec 247.9 s/step gen (flat) vs C8 DAS **183.5 avg (-26%)**: -17%
in the cold first epoch rising to **-28% steady-state** (best steps -37%)
as trees accumulate up to 80 trajectories/problem — inside the DAS
paper's 25-50% band, at T=1.0 on a non-distilled model. Acceptance 16.5%
(8.4 drafted / 1.38 accepted per round; ceiling 24 never binds — the
match-length cap and min_token_prob govern). Rewards equivalent (0.885
vs 0.906 mean; both arms approach ~1.0 on the repeated subset —
train-reward inflation from repetition, use held-out eval for learning
claims). KNOWN LEAK, fix designed: total step time barely moved because
~47s/step leaked back into training-phase timings — the pump's
engine-side sleep gate never fires on verl 0.9 colocated (verl sleeps
engines outside vLLMHttpServer.sleep), and tail_max_active=0 un-gates
mid-decode applies (knob coupling). Fix: service-side per-iteration
delta serving (get_deltas withholds until iteration advances) + separate
apply-gate knob; also shrink window_iterations to cap resident tree RAM
in worker processes.

### Benchmarking traps (all hit while producing these numbers)

1. **Logprobs silently disable speculation.** vLLM excludes any request
   with `logprobs` set (`is_spec_decode_unsupported`), and verl's agent
   loop requests them unconditionally. Every spec arm drafts *zero* tokens
   until stripped (`PYIS_STRIP_ROLLOUT_LOGPROBS=1`; safe when the trainer
   recomputes old log-probs, verl's default). Check
   `vllm:spec_decode_num_drafts > 0` before believing any spec benchmark.
2. **`/tmp/metrics` multiproc residue compounds across jobs.** Dead
   engines' prometheus files survive on the pod; every `/metrics` scrape
   aggregates all of them, and the scheduler scrapes per request. 600+
   dead files inflated the baseline from 46.8 to 70.5 s/step and grew
   every run. Now auto-cleaned at engine launch (dead-PID check in
   `VllmEnginePatch`); the old ladder's absolutes were unusable.
3. **Back-to-back jobs race GPU teardown.** Submitting within seconds of
   the prior job's exit hits verl's resource check or CUDA OOM during
   engine init. Sequencers settle 180s between jobs.
4. **Replacement pods come up blank.** GKE pod churn silently removes
   verl (editable at /tmp/verl), arctic, and datasets; jobs then die in
   60s before reaching wandb. Re-provision from a healthy sibling pod.
5. **Occupancy-gating drafting was a net loss here.** Decode at these
   batch sizes is memory-bound (roofline crossover ~batch 300 for a 4B
   model on H100), so drafting pays all step, not just in the straggler
   tail — `tail_max_active: 0` (always draft) is the right default; a
   positive gate is for compute-bound regimes only.
6. **Never let the data plane touch the decode loop.** Mid-decode
   `collective_rpc` (payloads *or* polling) costs several s/step; all
   tree delivery now happens in a fire-and-forget drain at the wake_up
   boundary, overlapping prefill.
