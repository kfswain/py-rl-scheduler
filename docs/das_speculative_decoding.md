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
