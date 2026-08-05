# Copyright 2026 llm-d
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DAS-enabled variant of run_qwen2_5-32b_math.sh. Deltas vs baseline:
#   - dataset: DeepScaleR competition math (long chains of thought — the
#     long-tail rollout regime DAS targets; same dataset family as the DAS
#     paper's math benchmark). Prepare once with:
#       python3 integration/verl/examples/prepare_deepscaler.py
#   - max_response_length raised to 4096 so the long tail can actually form
#     (micro-batch sizes may need halving on memory-constrained GPUs).
#   - speculative_config enables vLLM suffix decoding (requires
#     arctic-inference in the image / runtime env; see runtime-env-das.yaml,
#     which also sets PYIS_DAS_ENABLED=1 and DAS_CONFIG_PATH).
#   - num_speculative_tokens must be >= the Long budget in das_config.yaml.
#   - engine_kwargs passthrough is verified on verl 0.9.x; on verl 0.7.x,
#     if the override is rejected, remove it — DAS trajectory gathering
#     still runs, engines just decode without speculation.
#
# A/B guide (10-step runs, compare rollout time per step + reward curve):
#   arm A: this script minus the speculative_config line  (no spec decode)
#   arm B: this script with PYIS_DAS_ENABLED=0            (stock suffix)
#   arm C: this script                                    (DAS)

set -x

deepscaler_train_path=/home/ray/data/deepscaler/train.parquet
deepscaler_test_path=/home/ray/data/deepscaler/test.parquet

train_files="['$deepscaler_train_path']"
test_files="['$deepscaler_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-32B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_32b_math' \
    trainer.experiment_name='qwen32b_deepscaler_das' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_training_steps=10 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config={"method": "suffix", "num_speculative_tokens": 24, "suffix_decoding_max_tree_depth": 24, "suffix_decoding_max_spec_factor": 2.0, "suffix_decoding_min_token_prob": 0.1}' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=integration.verl.verl_hook.PyInferenceAgentLoopManager $@
