#!/usr/bin/env bash
# Disaggregated (fully-async) DAS A/B run script.
#
# Topology: FSDP training on the H100 trainer pool (trainer.nnodes x 8),
# vLLM rollout on the H200 pool (rollout.nnodes x 8, standalone replicas —
# engines never sleep). Pool separation is enforced by accelerator_type
# bundle pinning; scheduler routing + DAS ride in via
# integration.verl.fully_async_das.setup (worker_process_setup_hook).
#
# Required runtime-env (see sequence9.sh):
#   worker_process_setup_hook: integration.verl.fully_async_das.setup
#   PYTHONPATH=.:./src  PYIS_STRIP_ROLLOUT_LOGPROBS=1
#   PYIS_TRAINER_ACCEL_TYPE=accelerator_type:H100
#   PYIS_ROLLOUT_ACCEL_TYPE=accelerator_type:H200
#   (+ PYIS_DAS_ENABLED=1, DAS_CONFIG_PATH=./configs/das_config.yaml for C arms)
#
# Notes vs the colocated (sync) scripts:
# - calculate_log_probs=True satisfies the fully-async rollouter assert;
#   PYIS_STRIP_ROLLOUT_LOGPROBS=1 still strips logprobs at the engine
#   request (spec-decode eligibility). The trainer recomputes old_log_prob
#   because rollout_correction.bypass_mode=False (decoupled mode).
# - trigger_parameter_sync_step=1 + staleness_threshold=0: weight sync every
#   trainer step, at most one batch generated ahead (one-step-off pipeline).
# - require_batches x ppo_mini_batch_size = 32 prompts/step (x n=8 = 256
#   trajectories), matching the colocated A5/C8 step size.
# - trainer.nnodes=2 (16 H100 GPUs): verl's sequence balancer requires
#   trajectories-per-step % trainer world size == 0 (256 % 24 != 0), and 16
#   matches the colocated pair's DP width exactly. The third H100 node
#   stays idle.
# - fsdp2 is REQUIRED: the fully-async trainer's save/restore_model_to_cpu
#   (version-1 params for the old_log_prob recompute) asserts DTensor
#   parameters. Differs from the colocated pair's fsdp1 — shared by both
#   arms here, so the A/C comparison is unaffected.
set -x
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    data.train_files=/home/ray/data/deepmath/train.parquet \
    data.val_files=/home/ray/data/deepmath/test.parquet \
    data.return_raw_chat=True \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.train_max_samples=128 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    rollout.nnodes=2 \
    rollout.n_gpus_per_node=8 \
    rollout.n=8 \
    rollout.total_rollout_steps=1280 \
    async_training.staleness_threshold=0 \
    async_training.trigger_parameter_sync_step=1 \
    async_training.require_batches=2 \
    async_training.partial_rollout=True \
    async_training.use_trainer_do_validate=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='das_deepscaler_ab' \
    trainer.experiment_name='armA7_32b_deepmath_disagg' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=10 \
    $@
