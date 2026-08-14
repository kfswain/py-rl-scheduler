#!/usr/bin/env bash
# Colocated (sync-trainer) DAS/spec-arm run script — the base for the
# A2..C8 benchmark ladders (see docs/das_speculative_decoding.md).
#
# Historically hand-authored on the Ray head at /tmp/das_repo/run_das_wandb.sh
# and lost to the 2026-08-10 head-pod eviction; restored here verbatim so the
# docs' repro commands have a durable source. Baked-in values are the 4B
# DeepScaleR smoke defaults; real arms override via appended Hydra args
# (last-wins), e.g. the C8 launch in the docs.
#
# The scheduler hook is injected via the agent_loop_manager_class override
# below — the sync trainer honors it; the disaggregated (fully-async) runs
# use run_das_disagg.sh + worker_process_setup_hook instead.
set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/ray/data/deepscaler/train.parquet \
    data.val_files=/home/ray/data/deepscaler/test.parquet \
    data.return_raw_chat=True \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=ngram \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens=24 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.prompt_lookup_max=8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.prompt_lookup_min=2 \
    '+actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager' \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='das_deepscaler_ab' \
    trainer.experiment_name='qwen3_4b_deepscaler_das_ngram' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_training_steps=3 \
    $@
