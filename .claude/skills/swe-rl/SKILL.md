---
name: swe-rl
description: Operate the SWE-bench RL training pipeline (verl + GKE agent-sandbox + py-inference-scheduler): run preflight validation, launch/monitor training jobs, recover from known failures, backfill W&B. Use when asked to run, check, debug, or resume SWE RL training on the GKE cluster.
---

# SWE RL Pipeline Operations

Full background: `docs/swe_bench_guide.md` (architecture, phases, every failure
mode discovered during bring-up). This skill is the operational fast path.

## 1. Preflight — always run before launching

```bash
uv run --with pandas --with pyarrow python -m integration.verl.swe.preflight
```

From the repo root, against the current kubectl context. Validates: sandbox
CRDs + gVisor pool + policy-exempt namespace + RBAC; a live sandbox
create→exec→delete smoke test with a real task image (proves mirror
pull-through and admission-policy shape); verl importable at ONE commit on
every Ray pod; W&B key; parquets on EVERY replica with matching rows and full
schema; stray sandboxes; disk. Exit 0 = clear to launch.

Remediations by check name:

| Failing check | Fix |
|---|---|
| sandbox CRD / gVisor pool | Install agent-sandbox on GKE (guide Phase 0 link); pool needs `sandbox.gke.io/runtime=gvisor` nodes |
| RBAC | `kubectl apply -f configs/swe_sandbox_rbac.yaml` |
| verl on all ray pods | A replaced worker pod is blank (GKE node repairs do this). Provision: `kubectl exec <pod> -- sh -c 'mkdir -p /tmp/verl && cd /tmp/verl && git clone https://github.com/volcengine/verl.git && cd verl && git checkout <COMMIT_FROM_HEALTHY_POD> && pip install -e . --no-deps'` |
| parquet missing / row mismatch | Data is per-pod (emptyDir). Redistribute: `kubectl exec <head> -- tar cf - -C /home/ray/data swe \| kubectl exec -i <pod> -- tar xf - -C /home/ray/data` |
| parquet schema | Rerun `integration/verl/examples/prepare_swe_dataset.py` on the head pod with the `--registry_rewrite` flags (guide Phase 0), then re-filter with the calibration keep-list and redistribute |
| sandbox smoke: pod Running | Pool capacity (check autoscaling + stray sandboxes), image pull (mirror cache), or admission policy changes |
| stray sandboxes | `kubectl get sandboxes -n agents-system` → delete `swe-*`/`grade-*`/`cal-*` leftovers |

## 2. Launch

Port-forward once per session:
`kubectl port-forward svc/<ray-head-svc> 8265:8265 &`

```bash
uv run ray job submit --address http://127.0.0.1:8265 --no-wait \
    --runtime-env integration/verl/examples/runtime-env-swe.yaml \
    -- bash integration/verl/examples/run_swe.sh \
    data.train_files=/home/ray/data/swe/train_calibrated.parquet \
    data.train_batch_size=<B> \
    actor_rollout_ref.rollout.n=<N> \
    actor_rollout_ref.actor.ppo_mini_batch_size=<B> \
    trainer.total_training_steps=<STEPS> \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.experiment_name=<UNIQUE_NAME>
```

- Submit from the repo root (the runtime env ships the working tree).
- **Sandbox concurrency = B × N + grading burst; keep it under the pool
  ceiling** (~3 sandboxes per e2-standard-2 node × max autoscaled nodes).
- Validation on the 500 SWE-bench eval rows creates 500 sandboxes at once —
  keep `val_before_train=False` / `test_freq=-1` unless the pool is sized for it.
- Always train on `train_calibrated.parquet` (calibration-filtered), never raw.
- Easier-first curriculum: filter rows by `extra_info.num_non_test_lines <= 10`
  into a new parquet (median gold fix is 10 lines; ~50% of the set).
- W&B project is set in `run_swe.sh` (`trainer.project_name`); experiment name
  must be unique per launch.
- Scheduler A/B treatment arm: append
  `+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager`

## 3. Monitor

```bash
RAY_ADDRESS=http://127.0.0.1:8265 uv run ray job status <JOB_ID>
RAY_ADDRESS=http://127.0.0.1:8265 uv run ray job logs <JOB_ID> | grep -oE "step:[0-9]+ .*" | tail -1
kubectl get pods -n agents-system | grep -c swe-   # live rollout sandboxes
```

Healthy signatures (from validated runs): `num_turns/mean` 35–50;
`tool_calls/mean` >> `generate_sequences/mean` (sandbox-bound is normal);
`rollout_probs_pearson_corr` ≈ 0.999 (token discipline intact);
`critic/score/mean` mostly 0 at 7B scale — `score/max` spiking to 1.0 is a
solve. Startup takes ~10 min (model load + vLLM) before the W&B run or any
sandbox appears; steps are ~10–20 min tail-bound.

## 4. Post-run

- **W&B history may be partial or empty** (SDK→service channel dies during
  long idle gaps). Backfill from console logs — metrics are always complete
  there: save job logs to a file, `kubectl cp` it + `integration/verl/examples/backfill_wandb.py`
  to the head pod (it has `WANDB_API_KEY`), then
  `python3 backfill_wandb.py --log_file <log> --project <project> --run_id <id>`.
- Checkpoints land in `checkpoints/<project>/<experiment>/` on the driver pod.
- Re-run preflight afterwards; it flags leaked sandboxes.

## 5. Known failure modes (all previously hit)

- **Job hangs in init / dies with "node terminated"**: a GPU node is dying or
  died. Check `ray status` recent failures + `kubectl get pods`. After GKE
  auto-repair, the replacement worker pod is BLANK → run preflight, fix
  `verl on all ray pods` + parquet checks before resubmitting.
- **`ModuleNotFoundError: No module named 'verl'`**: blank replacement pod
  (above).
- **`FileNotFoundError: .../train*.parquet`**: TaskRunner landed on a pod
  without data — redistribute (table above).
- **All rewards 0.0 with `swe_reason: rollout-error`**: sandbox plumbing.
  Check pool capacity/autoscaling, stray sandboxes, and RBAC; per-trajectory
  reasons are in `extra_fields.swe_reason`.
- **Sandboxes Pending until timeout**: pool contention (another sweep/job?) or
  autoscaler at max. `SWE_SANDBOX_WAIT_S` (default 600) covers scale-up.
- **Writing files into sandboxes**: exec+base64 only — `kubectl cp` exits
  nonzero (dropped CAP_CHOWN) even though content lands.
- **Calibration**: rerun `integration.verl.swe.calibrate` after dataset or
  grader changes; investigate `key-mismatch` clusters before dropping
  instances (two R2E spec-key conventions were mapping bugs on our side).
