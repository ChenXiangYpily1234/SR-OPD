# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Cost Meter for tracking teacher vLLM / reward_model call costs.

Tracks:
- Teacher call count, latency, sequence count, batch size
- Token counts (prompt, response, supervised, scored)
- GPU memory usage
- Throughput metrics
- Saving / reduction ratios
"""

import time
import logging
import warnings
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CostMeter:
    """Tracks teacher call costs and training token statistics."""

    def __init__(
        self,
        enabled: bool = True,
        track_timing: bool = True,
        track_gpu_memory: bool = True,
        track_teacher_call: bool = True,
        warn_once: bool = True,
    ):
        self.enabled = enabled
        self.track_timing = track_timing
        self.track_gpu_memory = track_gpu_memory
        self.track_teacher_call = track_teacher_call
        self.warn_once = warn_once
        self._warned = set()

        self.reset_step()

        # Total accumulators
        self._total_elapsed_time_sec = 0.0
        self._total_teacher_call_count = 0
        self._total_teacher_sequence_count = 0
        self._total_teacher_processed_input_token_count = 0
        self._total_teacher_scored_response_token_count = 0
        self._total_teacher_returned_logprob_count = 0
        self._total_supervised_token_count = 0
        self._total_valid_response_token_count = 0
        self._step_start_time = time.time()

        self._timers = {}

    def _warn(self, msg):
        if self.warn_once and msg in self._warned:
            return
        self._warned.add(msg)
        warnings.warn(msg)

    def reset_step(self):
        """Reset per-step accumulators."""
        self._step_teacher_call_count = 0
        self._step_teacher_call_type = "unknown"
        self._step_teacher_sequence_count = 0
        self._step_teacher_batch_size = 0
        self._step_teacher_latency_sec = 0.0

        self._step_prompt_token_count = 0
        self._step_response_token_count = 0
        self._step_valid_response_token_count = 0
        self._step_supervised_token_count = 0
        self._step_teacher_processed_input_token_count = 0
        self._step_teacher_scored_response_token_count = 0
        self._step_teacher_returned_logprob_count = 0
        self._step_expected_logprobs_per_scored_token = None

        self._step_full_query_teacher_processed_input_token_count = 0
        self._step_full_query_teacher_scored_response_token_count = 0
        self._step_full_query_teacher_call_count = 0

        self._timers = {}

    def add_teacher_call(
        self,
        call_count=1,
        call_type="unknown",
        sequence_count=0,
        batch_size=0,
        prompt_token_count=0,
        response_token_count=0,
        processed_input_token_count=0,
        scored_response_token_count=0,
        returned_logprob_count=0,
        expected_logprobs_per_scored_token=None,
        latency_sec=0.0,
    ):
        if not self.enabled or not self.track_teacher_call:
            return
        self._step_teacher_call_count += call_count
        self._step_teacher_call_type = call_type
        self._step_teacher_sequence_count += sequence_count
        self._step_teacher_batch_size += batch_size
        self._step_teacher_latency_sec += latency_sec
        self._step_teacher_processed_input_token_count += processed_input_token_count
        self._step_teacher_scored_response_token_count += scored_response_token_count
        self._step_teacher_returned_logprob_count += returned_logprob_count
        if expected_logprobs_per_scored_token is not None:
            expected = int(expected_logprobs_per_scored_token)
            if expected < 1:
                raise ValueError("expected_logprobs_per_scored_token must be positive")
            if (
                self._step_expected_logprobs_per_scored_token is not None
                and self._step_expected_logprobs_per_scored_token != expected
            ):
                # Different query widths can coexist in one step. Mark the
                # step as mixed-width; each call's returned count is still
                # recorded, but a single K invariant cannot describe the
                # combined calls.
                self._step_expected_logprobs_per_scored_token = 0
            elif self._step_expected_logprobs_per_scored_token is None:
                self._step_expected_logprobs_per_scored_token = expected
        self._step_prompt_token_count += prompt_token_count
        self._step_response_token_count += response_token_count

    def add_tokens(
        self,
        prompt_token_count=0,
        response_token_count=0,
        valid_response_token_count=0,
        supervised_token_count=0,
        teacher_processed_input_token_count=0,
        teacher_scored_response_token_count=0,
        full_query_teacher_processed_input_token_count=0,
        full_query_teacher_scored_response_token_count=0,
        full_query_teacher_call_count=0,
    ):
        if not self.enabled:
            return
        self._step_prompt_token_count += prompt_token_count
        self._step_response_token_count += response_token_count
        self._step_valid_response_token_count += valid_response_token_count
        self._step_supervised_token_count += supervised_token_count
        if teacher_processed_input_token_count > 0:
            self._step_teacher_processed_input_token_count += teacher_processed_input_token_count
        if teacher_scored_response_token_count > 0:
            self._step_teacher_scored_response_token_count += teacher_scored_response_token_count
        if full_query_teacher_processed_input_token_count > 0:
            self._step_full_query_teacher_processed_input_token_count += full_query_teacher_processed_input_token_count
        if full_query_teacher_scored_response_token_count > 0:
            self._step_full_query_teacher_scored_response_token_count += full_query_teacher_scored_response_token_count
        if full_query_teacher_call_count > 0:
            self._step_full_query_teacher_call_count += full_query_teacher_call_count

    def timer(self, name):
        return _TimerContext(self, name)

    def get_timer(self, name):
        return self._timers.get(name, 0.0)

    def _get_gpu_memory_metrics(self):
        metrics = {}
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                metrics["cost/gpu_allocated_mem_gb_rank0"] = allocated
                metrics["cost/gpu_reserved_mem_gb_rank0"] = reserved
                metrics["cost/gpu_max_allocated_mem_gb_rank0"] = max_allocated
                metrics["cost/gpu_max_reserved_mem_gb_rank0"] = max_reserved

                if torch.distributed.is_initialized():
                    try:
                        alloc_t = torch.tensor([allocated], device="cuda")
                        torch.distributed.all_reduce(alloc_t, op=torch.distributed.ReduceOp.AVG)
                        metrics["cost/gpu_allocated_mem_gb_mean"] = alloc_t.item()
                        metrics["cost/gpu_allocated_mem_gb_max"] = alloc_t.item()
                    except Exception:
                        pass
        except Exception:
            self._warn("Failed to get GPU memory metrics")
        return metrics

    def get_step_metrics(self):
        if not self.enabled:
            return {}
        metrics = {}

        if (
            self._step_expected_logprobs_per_scored_token is not None
            and self._step_expected_logprobs_per_scored_token > 0
            and self._step_teacher_scored_response_token_count > 0
        ):
            expected_logprob_count = (
                self._step_teacher_scored_response_token_count
                * self._step_expected_logprobs_per_scored_token
            )
            if self._step_teacher_returned_logprob_count != expected_logprob_count:
                raise AssertionError(
                    "teacher returned logprob count does not match scored response tokens * top-k: "
                    f"{self._step_teacher_returned_logprob_count} != "
                    f"{self._step_teacher_scored_response_token_count} * "
                    f"{self._step_expected_logprobs_per_scored_token}"
                )

        if self._step_teacher_call_count > 0:
            metrics["cost/teacher_call_count"] = self._step_teacher_call_count
            metrics["cost/teacher_sequence_count"] = self._step_teacher_sequence_count
            metrics["cost/teacher_batch_size"] = self._step_teacher_batch_size
            metrics["cost/teacher_latency_sec"] = self._step_teacher_latency_sec
            metrics["cost/teacher_avg_latency_sec"] = (
                self._step_teacher_latency_sec / max(self._step_teacher_call_count, 1)
            )
            call_type = self._step_teacher_call_type
            if call_type == "vllm":
                metrics["cost/teacher_vllm_call_count"] = self._step_teacher_call_count
                metrics["cost/teacher_vllm_sequence_count"] = self._step_teacher_sequence_count
                metrics["cost/teacher_vllm_latency_sec"] = self._step_teacher_latency_sec
                metrics["cost/teacher_vllm_avg_latency_sec"] = metrics["cost/teacher_avg_latency_sec"]
            elif call_type in ("reward_model_forward", "reward_model"):
                metrics["cost/teacher_forward_call_count"] = self._step_teacher_call_count
                metrics["cost/teacher_forward_sequence_count"] = self._step_teacher_sequence_count
                metrics["cost/teacher_forward_latency_sec"] = self._step_teacher_latency_sec
                metrics["cost/teacher_forward_avg_latency_sec"] = metrics["cost/teacher_avg_latency_sec"]
            call_type_id = {"unknown": 0, "vllm": 1, "reward_model_forward": 2, "reward_model": 2}
            metrics["cost/teacher_call_type_id"] = call_type_id.get(call_type, 0)

        metrics["cost/teacher_prompt_token_count"] = self._step_prompt_token_count
        metrics["cost/teacher_response_token_count"] = self._step_response_token_count
        metrics["cost/teacher_processed_input_token_count"] = self._step_teacher_processed_input_token_count
        metrics["cost/teacher_scored_response_token_count"] = self._step_teacher_scored_response_token_count
        metrics["cost/teacher_returned_logprob_count"] = self._step_teacher_returned_logprob_count
        metrics["cost/student_generated_response_token_count"] = self._step_valid_response_token_count
        metrics["cost/final_supervised_token_count"] = self._step_supervised_token_count

        if self._step_teacher_latency_sec > 0:
            metrics["cost/teacher_processed_tokens_per_sec"] = (
                self._step_teacher_processed_input_token_count / self._step_teacher_latency_sec
            )
            metrics["cost/teacher_scored_tokens_per_sec"] = (
                self._step_teacher_scored_response_token_count / self._step_teacher_latency_sec
            )

        if self._step_full_query_teacher_scored_response_token_count > 0:
            metrics["cost/teacher_scored_response_token_saving_ratio"] = max(
                0.0,
                1.0 - self._step_teacher_scored_response_token_count / max(
                    self._step_full_query_teacher_scored_response_token_count, 1
                ),
            )
        if self._step_full_query_teacher_processed_input_token_count > 0:
            metrics["cost/teacher_processed_input_token_saving_ratio"] = max(
                0.0,
                1.0 - self._step_teacher_processed_input_token_count / max(
                    self._step_full_query_teacher_processed_input_token_count, 1
                ),
            )
        if self._step_full_query_teacher_call_count > 0:
            metrics["cost/teacher_call_saving_ratio"] = max(
                0.0,
                1.0 - self._step_teacher_call_count / max(self._step_full_query_teacher_call_count, 1),
            )

        if self._step_valid_response_token_count > 0:
            metrics["cost/global_supervised_token_ratio"] = (
                self._step_supervised_token_count / self._step_valid_response_token_count
            )
            metrics["cost/teacher_query_token_ratio"] = (
                self._step_teacher_scored_response_token_count
                / self._step_valid_response_token_count
            )
            metrics["cost/supervised_token_reduction_ratio"] = max(
                0.0,
                1.0 - self._step_supervised_token_count / max(self._step_valid_response_token_count, 1),
            )
            metrics["cost/supervised_over_teacher_scored_ratio"] = (
                self._step_supervised_token_count / max(self._step_teacher_scored_response_token_count, 1)
            )
            metrics["cost/supervised_over_teacher_processed_ratio"] = (
                self._step_supervised_token_count / max(self._step_teacher_processed_input_token_count, 1)
            )

        if self.track_gpu_memory:
            metrics.update(self._get_gpu_memory_metrics())

        # Update totals
        self._total_teacher_call_count += self._step_teacher_call_count
        self._total_teacher_sequence_count += self._step_teacher_sequence_count
        self._total_teacher_processed_input_token_count += self._step_teacher_processed_input_token_count
        self._total_teacher_scored_response_token_count += self._step_teacher_scored_response_token_count
        self._total_teacher_returned_logprob_count += self._step_teacher_returned_logprob_count
        self._total_supervised_token_count += self._step_supervised_token_count
        self._total_valid_response_token_count += self._step_valid_response_token_count

        return metrics

    def get_total_metrics(self):
        if not self.enabled:
            return {}
        self._total_elapsed_time_sec = time.time() - self._step_start_time
        metrics = {
            "cost_total/elapsed_time_sec": self._total_elapsed_time_sec,
            "cost_total/teacher_call_count": self._total_teacher_call_count,
            "cost_total/teacher_sequence_count": self._total_teacher_sequence_count,
            "cost_total/teacher_processed_input_token_count": self._total_teacher_processed_input_token_count,
            "cost_total/teacher_scored_response_token_count": self._total_teacher_scored_response_token_count,
            "cost_total/teacher_returned_logprob_count": self._total_teacher_returned_logprob_count,
            "cost_total/supervised_token_count": self._total_supervised_token_count,
            "cost_total/valid_response_token_count": self._total_valid_response_token_count,
        }
        try:
            import torch
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                gpu_hours = self._total_elapsed_time_sec * world_size / 3600.0
                metrics["cost_total/gpu_hours"] = gpu_hours
        except Exception:
            pass
        if self._total_valid_response_token_count > 0:
            metrics["cost_total/global_supervised_token_ratio"] = (
                self._total_supervised_token_count / self._total_valid_response_token_count
            )
            metrics["cost_total/teacher_query_token_ratio"] = (
                self._total_teacher_scored_response_token_count
                / self._total_valid_response_token_count
            )
            metrics["cost_total/supervised_token_reduction_ratio"] = max(
                0.0,
                1.0 - self._total_supervised_token_count / max(self._total_valid_response_token_count, 1),
            )
        if self._total_teacher_scored_response_token_count > 0:
            metrics["cost_total/supervised_over_teacher_scored_ratio"] = (
                self._total_supervised_token_count / max(self._total_teacher_scored_response_token_count, 1)
            )
        if self._total_teacher_processed_input_token_count > 0:
            metrics["cost_total/supervised_over_teacher_processed_ratio"] = (
                self._total_supervised_token_count / max(self._total_teacher_processed_input_token_count, 1)
            )
        return metrics


class _TimerContext:
    def __init__(self, cost_meter, name):
        self.cost_meter = cost_meter
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        self.cost_meter._timers[self.name] = elapsed
        return False
