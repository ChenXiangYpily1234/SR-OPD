# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "RolloutConfig",
]


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    max_tokens: Optional[int] = None
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)


@dataclass
class TraceConfig(BaseConfig):
    backend: Optional[str] = None
    token2text: bool = False


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format"}

    name: Optional[str] = MISSING
    mode: str = "sync"
    skip_tokenizer_init: bool = True

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = True
    n: int = 1

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384
    log_prob_top_k: int = 256
    top_k_strategy: str = "only_stu"  # "only_stu", "only_tch", "intersection", or "union"
    reward_weight_mode: str = "student_p"  # "student_p", "teacher_p", or "none"
    teacher_temperature: float = 1.0  # Temperature for teacher logits (default 1.0, no scaling)

    # Teacher target representation used by OPD variants.
    opd_target_mode: str = "sampled_token"

    # TA-OPD configuration
    ta_opd_enable: bool = False
    ta_opd_mode: str = "teachability"  # "teachability", "compatibility", "disagreement", "entropy", "tip", "random"
    ta_opd_topk: int = 16
    ta_opd_score_top_k_strategy: str = "union"
    ta_opd_retain_ratio: float = 0.10
    ta_opd_retain_token_count: Optional[int] = None
    ta_opd_normalize_scope: str = "batch"
    ta_opd_normalize_by: str = "selected_tokens"  # "selected_tokens", "all_tokens", "original"
    ta_opd_eps: float = 1.0e-8
    ta_opd_use_full_logits: str = "auto"  # "auto", "true", "false"
    ta_opd_fallback: str = "topk_overlap"
    ta_opd_detach_score: bool = True
    ta_opd_log_stats: bool = True
    ta_opd_assert_teacher_full_query: bool = True
    ta_opd_warn_if_no_topk: bool = True

    opd_metrics_dir: Optional[str] = None
    opd_metrics_dump_interval: int = 10

    # Cost metrics configuration
    cost_metrics_enable: bool = True
    cost_metrics_track_timing: bool = True
    cost_metrics_track_gpu_memory: bool = True
    cost_metrics_track_teacher_call: bool = True
    cost_metrics_log_interval: int = 1
    cost_metrics_warn_once: bool = True

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    update_weights_bucket_megabytes: int = 512

    skip_rollout: bool = False

    skip_dump_dir: str = "/tmp/rollout_dump"

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    seed: int = 42

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = False

    def __post_init__(self):
        """Validate the rollout config"""
        if self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )

        if self.opd_metrics_dump_interval < 1:
            raise ValueError("opd_metrics_dump_interval must be at least 1")
        if self.ta_opd_enable and self.ta_opd_normalize_scope != "batch":
            raise ValueError("Paper TA-OPD requires ta_opd_normalize_scope=batch")
        if self.ta_opd_enable and self.ta_opd_topk <= 0:
            raise ValueError("Paper TA-OPD requires ta_opd_topk > 0 for selector scoring")
        if self.ta_opd_enable and self.ta_opd_score_top_k_strategy != "union":
            raise ValueError("Paper TA-OPD requires ta_opd_score_top_k_strategy=union")
