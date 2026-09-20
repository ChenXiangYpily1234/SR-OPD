# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import math
import os
import time

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.boundary_opd import (
    BoundaryCaptureTarget,
    BoundaryOPDSettings,
    boundary_state_count,
    build_boundary_indices,
    build_uniform_response_state_indices,
    extract_hidden_tensor,
    gather_boundary_states_from_hidden,
    gather_boundary_states_from_sequence_shard,
    resolve_boundary_capture_target,
)
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.ht_opd import aggregate_ht_opd_numerator
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.ta_opd import select_opd_training_reward
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self._prior_opd_warnings = set()
        self._boundary_opd_warnings = set()



    def _warn_prior_opd_once(self, key: str, message: str) -> None:
        if key in self._prior_opd_warnings:
            return
        self._prior_opd_warnings.add(key)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.warning(message)

    def _warn_boundary_opd_once(self, key: str, message: str) -> None:
        if key in self._boundary_opd_warnings:
            return
        self._boundary_opd_warnings.add(key)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.warning(message)

    def _boundary_hidden_size(self, target: BoundaryCaptureTarget | None) -> int:
        module = self.actor_module
        visited = set()
        while id(module) not in visited:
            visited.add(id(module))
            next_module = None
            for attribute in ("_fsdp_wrapped_module", "module"):
                candidate = getattr(module, attribute, None)
                if isinstance(candidate, nn.Module) and candidate is not module:
                    next_module = candidate
                    break
            if next_module is None:
                break
            module = next_module

        config = getattr(module, "config", None)
        for candidate_config in (config, getattr(config, "text_config", None)):
            if candidate_config is None:
                continue
            for attribute in ("hidden_size", "n_embd", "d_model"):
                value = getattr(candidate_config, attribute, None)
                if value is not None:
                    return int(value)

        if target is not None:
            for attribute in ("in_features",):
                value = getattr(target.module, attribute, None)
                if value is not None:
                    return int(value)
            normalized_shape = getattr(target.module, "normalized_shape", None)
            if normalized_shape:
                return int(normalized_shape[-1])
            weight = getattr(target.module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim:
                return int(weight.shape[-1])
        raise RuntimeError("Boundary-OPD could not determine the actor hidden dimension")

    def _boundary_synchronize(self, device: torch.device) -> None:
        device_backend = getattr(torch, self.device_name, None)
        synchronize = getattr(device_backend, "synchronize", None)
        if not callable(synchronize):
            return
        try:
            synchronize(device)
        except TypeError:
            synchronize()

    def _boundary_memory_allocated(self, device: torch.device) -> int:
        device_backend = getattr(torch, self.device_name, None)
        memory_allocated = getattr(device_backend, "memory_allocated", None)
        if not callable(memory_allocated):
            return 0
        try:
            return int(memory_allocated(device))
        except TypeError:
            return int(memory_allocated())

    def _forward_with_boundary_capture(
        self,
        *,
        forward_call,
        settings: BoundaryOPDSettings | None,
        capture_target: BoundaryCaptureTarget | None,
        capture_resolution_error: Exception | None,
        state_indices: torch.Tensor | None,
        transition_valid_mask: torch.Tensor | None,
        batch_size: int,
        sequence_length: int,
        hidden_dim: int | None,
        prompt_ids=None,
        unpadded_indices: torch.Tensor | None = None,
        sequence_padding_size: int = 0,
    ):
        if settings is None:
            return forward_call(), None

        assert state_indices is not None
        assert transition_valid_mask is not None
        assert hidden_dim is not None
        capture_result = {
            "states": None,
            "capture_time_ms": 0.0,
            "communication_time_ms": 0.0,
            "extra_peak_memory_mb": 0.0,
            "error_code": 0,
            "error": capture_resolution_error,
            "hook_calls": 0,
            "captured_hidden_shape": None,
        }

        def capture_hidden(value) -> None:
            capture_result["hook_calls"] += 1
            if capture_result["hook_calls"] != 1:
                capture_result["error_code"] = 3
                capture_result["error"] = RuntimeError("Boundary-OPD capture hook ran more than once")
                capture_result["states"] = None
                return
            try:
                hidden = extract_hidden_tensor(value)
                if hidden is None:
                    raise RuntimeError("Boundary-OPD capture hook did not receive a hidden tensor")
                capture_result["captured_hidden_shape"] = tuple(hidden.shape)
                self._boundary_synchronize(hidden.device)
                capture_start = time.perf_counter()
                memory_before = self._boundary_memory_allocated(hidden.device)
                if self.use_remove_padding and self.use_ulysses_sp:
                    process_group = get_ulysses_sequence_parallel_group()
                    if process_group is None:
                        raise RuntimeError("Boundary-OPD cannot find the Ulysses sequence-parallel group")
                    communication_start = time.perf_counter()
                    states = gather_boundary_states_from_sequence_shard(
                        hidden,
                        state_indices,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        unpadded_indices=unpadded_indices,
                        sequence_padding_size=sequence_padding_size,
                        sequence_parallel_size=self.ulysses_sequence_parallel_size,
                        sequence_parallel_rank=torch.distributed.get_rank(process_group),
                        process_group=process_group,
                        hidden_storage_dtype=settings.storage_dtype,
                    )
                    self._boundary_synchronize(hidden.device)
                    capture_result["communication_time_ms"] = (
                        time.perf_counter() - communication_start
                    ) * 1000.0
                else:
                    states = gather_boundary_states_from_hidden(
                        hidden,
                        state_indices,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        hidden_storage_dtype=settings.storage_dtype,
                        unpadded_indices=unpadded_indices,
                    )
                    self._boundary_synchronize(hidden.device)
                memory_after = self._boundary_memory_allocated(hidden.device)
                retained_bytes = states.numel() * states.element_size()
                # This is an observable lower bound: the retained sparse tensor
                # bytes or the allocation delta seen across the hook, whichever
                # is larger. It does not claim to reset/read a global peak.
                capture_result["extra_peak_memory_mb"] = max(
                    memory_after - memory_before,
                    retained_bytes,
                ) / (1024.0 * 1024.0)
                capture_result["capture_time_ms"] = (time.perf_counter() - capture_start) * 1000.0
                capture_result["states"] = states
            except Exception as error:
                capture_result["error_code"] = 3
                capture_result["error"] = error

        hook_handle = None
        if capture_target is None:
            capture_result["error_code"] = 1
        elif capture_target.capture_method == "final_norm_forward_hook":
            hook_handle = capture_target.module.register_forward_hook(
                lambda _module, _inputs, output: capture_hidden(output)
            )
        elif capture_target.capture_method == "lm_head_forward_pre_hook":
            hook_handle = capture_target.module.register_forward_pre_hook(
                lambda _module, inputs: capture_hidden(inputs)
            )
        else:
            capture_result["error_code"] = 1
            capture_result["error"] = RuntimeError(
                f"unsupported Boundary-OPD capture method: {capture_target.capture_method}"
            )

        try:
            output = forward_call()
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if capture_result["states"] is None:
            if capture_result["error_code"] == 0:
                capture_result["error_code"] = 2
                capture_result["error"] = RuntimeError("Boundary-OPD capture hook was not invoked")
            error = capture_result["error"]
            if not settings.fallback_to_ff_cost:
                formatted_prompt_ids = (
                    [str(prompt_id) for prompt_id in prompt_ids]
                    if prompt_ids is not None
                    else ["<unknown>"]
                )
                raise RuntimeError(
                    "Boundary-OPD hidden capture failed with fallback_to_ff_cost=false: "
                    f"prompt_id={formatted_prompt_ids}, "
                    f"input_ids_shape={(batch_size, sequence_length)}, "
                    f"state_indices_shape={tuple(state_indices.shape)}, "
                    f"transition_valid_mask_shape={tuple(transition_valid_mask.shape)}, "
                    f"captured_hidden_shape={capture_result['captured_hidden_shape']}, "
                    f"unpadded_indices_shape="
                    f"{tuple(unpadded_indices.shape) if unpadded_indices is not None else None}, "
                    f"expected_boundary_states_shape="
                    f"{(batch_size, state_indices.shape[1], hidden_dim)}, "
                    f"reason={type(error).__name__}: {error}"
                ) from error
            self._warn_boundary_opd_once(
                f"capture_failure_{capture_result['error_code']}",
                "Boundary-OPD hidden capture failed; falling back to FF cost. "
                f"error_code={capture_result['error_code']}, error={error}",
            )
            capture_result["states"] = torch.zeros(
                batch_size,
                state_indices.shape[1],
                hidden_dim,
                device=state_indices.device,
                dtype=settings.storage_dtype,
            )
            transition_valid_mask = torch.zeros_like(transition_valid_mask, dtype=torch.bool)
            capture_success = False
        else:
            if capture_result["states"].shape[-1] != hidden_dim:
                raise AssertionError(
                    "Boundary-OPD captured hidden width differs from the actor configuration: "
                    f"{capture_result['states'].shape[-1]} vs {hidden_dim}"
                )
            capture_success = True

        device = capture_result["states"].device
        batch_float = lambda value: torch.full(
            (batch_size,),
            float(value),
            dtype=torch.float32,
            device=device,
        )
        boundary_outputs = (
            capture_result["states"].detach(),
            transition_valid_mask.to(device=device, dtype=torch.bool).detach(),
            torch.full((batch_size,), capture_success, dtype=torch.bool, device=device),
            batch_float(capture_result["capture_time_ms"]),
            batch_float(capture_result["communication_time_ms"]),
            batch_float(capture_result["extra_peak_memory_mb"]),
            torch.full(
                (batch_size,),
                int(capture_result["error_code"]),
                dtype=torch.int64,
                device=device,
            ),
        )
        return output, boundary_outputs
    def _prior_opd_hidden_size(self) -> int:
        module = getattr(self.actor_module, "module", self.actor_module)
        module = getattr(module, "_fsdp_wrapped_module", module)
        config = getattr(module, "config", None)
        return int(getattr(config, "hidden_size", 1))
    def _pool_prior_opd_features(
        self,
        *,
        response_hidden: torch.Tensor | None,
        response_mask: torch.Tensor,
        responses: torch.Tensor,
        attention_mask: torch.Tensor,
        entropy: torch.Tensor | None,
        log_probs: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_log_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool student-only rollout features without retaining token-level hidden states."""
        response_mask = response_mask.bool()
        batch_size, response_length = response_mask.shape
        valid_lengths = response_mask.sum(dim=-1)

        if response_hidden is None:
            self._warn_prior_opd_once(
                "hidden_unavailable",
                "PRIOR-OPD could not read the actor's last hidden state; hidden features are zero-filled.",
            )
            response_hidden = torch.zeros(
                batch_size,
                response_length,
                self._prior_opd_hidden_size(),
                device=responses.device,
                dtype=torch.float32,
            )
        else:
            response_hidden = response_hidden.detach()

        hidden_mask = response_mask.unsqueeze(-1).to(response_hidden.dtype)
        hidden_denom = hidden_mask.sum(dim=1).clamp_min(1.0)
        hidden_mean = (response_hidden * hidden_mask).sum(dim=1) / hidden_denom

        nll = -log_probs.detach().float()
        entropy_f = torch.zeros_like(nll) if entropy is None else entropy.detach().float()
        uncertainty_logits = entropy_f + nll
        uncertainty_logits = uncertainty_logits.masked_fill(~response_mask, torch.finfo(uncertainty_logits.dtype).min)
        uncertainty_weights = torch.softmax(uncertainty_logits, dim=-1)
        uncertainty_weights = uncertainty_weights * response_mask.to(uncertainty_weights.dtype)
        uncertainty_weights = uncertainty_weights / uncertainty_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        hidden_uncertainty = (
            response_hidden * uncertainty_weights.to(response_hidden.dtype).unsqueeze(-1)
        ).sum(dim=1)

        response_positions = torch.arange(response_length, device=response_mask.device).expand(batch_size, -1)
        last_indices = response_positions.masked_fill(~response_mask, -1).max(dim=-1).values.clamp_min(0)
        gather_indices = last_indices.view(batch_size, 1, 1).expand(-1, 1, response_hidden.shape[-1])
        hidden_last = response_hidden.gather(dim=1, index=gather_indices).squeeze(1)
        hidden_last = hidden_last * (valid_lengths > 0).to(hidden_last.dtype).unsqueeze(-1)
        pooled_hidden = torch.stack([hidden_mean, hidden_uncertainty, hidden_last], dim=1).detach()

        prompt_lengths = attention_mask[:, :-response_length].sum(dim=-1).float()
        response_lengths = valid_lengths.float()
        truncated = (valid_lengths == response_length).float()

        repeat_ratio = torch.zeros(batch_size, device=responses.device, dtype=torch.float32)
        entropy_p90 = torch.zeros_like(repeat_ratio)
        nll_p90 = torch.zeros_like(repeat_ratio)
        for row in range(batch_size):
            row_mask = response_mask[row]
            row_tokens = responses[row][row_mask]
            if row_tokens.numel() > 0:
                repeat_ratio[row] = 1.0 - row_tokens.unique().numel() / row_tokens.numel()
                entropy_p90[row] = torch.quantile(entropy_f[row][row_mask], 0.9)
                nll_p90[row] = torch.quantile(nll[row][row_mask], 0.9)

        denom = response_lengths.clamp_min(1.0)
        entropy_mean = (entropy_f * response_mask).sum(dim=-1) / denom
        nll_mean = (nll * response_mask).sum(dim=-1) / denom

        margin_mean = torch.zeros_like(repeat_ratio)
        sampled_rank_mean = torch.zeros_like(repeat_ratio)
        if topk_ids is not None and topk_log_probs is not None and topk_ids.shape[-1] > 0:
            if topk_log_probs.shape[-1] >= 2:
                margin = (topk_log_probs[..., 0] - topk_log_probs[..., 1]).detach().float()
                margin_mean = (margin * response_mask).sum(dim=-1) / denom
            else:
                self._warn_prior_opd_once(
                    "topk_margin_unavailable",
                    "PRIOR-OPD margin_mean requires student top-k >= 2; the feature is zero-filled.",
                )

            sampled_matches = topk_ids.eq(responses.unsqueeze(-1))
            found = sampled_matches.any(dim=-1)
            sampled_rank = sampled_matches.float().argmax(dim=-1).float() + 1.0
            sampled_rank = torch.where(found, sampled_rank, torch.full_like(sampled_rank, topk_ids.shape[-1] + 1.0))
            sampled_rank_mean = (sampled_rank * response_mask).sum(dim=-1) / denom
        else:
            self._warn_prior_opd_once(
                "topk_features_unavailable",
                "PRIOR-OPD student top-k features are unavailable; margin and sampled-rank features are zero-filled.",
            )

        # The trainer overwrites the final column with normalized global progress.
        normalized_training_progress = torch.zeros_like(repeat_ratio)
        scalar_features = torch.stack(
            [
                torch.log1p(prompt_lengths),
                torch.log1p(response_lengths),
                truncated,
                repeat_ratio,
                entropy_mean,
                entropy_p90,
                nll_mean,
                nll_p90,
                margin_mean,
                sampled_rank_mean,
                normalized_training_progress,
            ],
            dim=-1,
        ).detach()
        return pooled_hidden, scalar_features
    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        top_k=0,
        student_top_k_ids=None,
        return_prior_opd_features=False,
        prior_opd_hidden_enable=True,
        boundary_opd_settings: BoundaryOPDSettings | None = None,
        boundary_capture_target: BoundaryCaptureTarget | None = None,
        boundary_capture_resolution_error: Exception | None = None,
        boundary_hidden_dim: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_ids: # (bs, response_len, k)
            topk_log_probs: # (bs, response_len, k)
            prior_opd_pooled_hidden: # (bs, 3, hidden_size), detached
            prior_opd_scalar_features: # (bs, 11), detached
            Boundary-OPD appends seven detached sparse-capture tensors when enabled.
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            topk_ids = None
            topk_log_probs = None
            response_hidden = None
            hidden_available = False
            boundary_outputs = None
            boundary_state_indices = None
            boundary_transition_valid_mask = None
            if boundary_opd_settings is not None:
                response_mask = micro_batch.get(
                    "boundary_response_mask",
                    micro_batch.get("response_mask", attention_mask[:, -response_length:]),
                )
                prompt_mask = attention_mask[:, :-response_length]
                if boundary_opd_settings.score_mode == "persistent_departure_area":
                    boundary_state_indices, boundary_transition_valid_mask = build_uniform_response_state_indices(
                        response_mask=response_mask,
                        prompt_mask=prompt_mask,
                        num_states=boundary_opd_settings.num_boundaries,
                    )
                else:
                    boundary_state_indices, boundary_transition_valid_mask = build_boundary_indices(
                        response_mask=response_mask,
                        prompt_mask=prompt_mask,
                        num_boundaries=boundary_opd_settings.num_boundaries,
                    )

            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                pad_size = 0
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if return_prior_opd_features and prior_opd_hidden_enable:
                    extra_args["output_hidden_states"] = True
                    extra_args["return_dict"] = True

                output, boundary_outputs = self._forward_with_boundary_capture(
                    forward_call=lambda: self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    ),
                    settings=boundary_opd_settings,
                    capture_target=boundary_capture_target,
                    capture_resolution_error=boundary_capture_resolution_error,
                    state_indices=boundary_state_indices,
                    transition_valid_mask=boundary_transition_valid_mask,
                    batch_size=batch_size,
                    sequence_length=seqlen,
                    hidden_dim=boundary_hidden_dim,
                    prompt_ids=micro_batch.get("ff_prompt_uid"),
                    unpadded_indices=indices,
                    sequence_padding_size=pad_size,
                )  # prevent model thinks we are generating

                if return_prior_opd_features and prior_opd_hidden_enable:
                    hidden_states = getattr(output, "hidden_states", None)
                    if hidden_states:
                        hidden_rmpad = hidden_states[-1].squeeze(0).detach()
                        hidden_available = True
                        # Release references to all earlier layers before the
                        # full-vocabulary log-softmax allocation below.
                        output.hidden_states = None
                        del hidden_states
                
                need_logits = top_k > 0

                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation and gradient
                    # issues from inplace operations
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        # Note: we don't use inplace_backward here to ensure correct gradients
                        # when both log_probs and topk_log_probs are needed
                        log_probs_all = torch.log_softmax(logits_rmpad, dim=-1)
                        # Gather log_probs for target tokens
                        log_probs = log_probs_all.gather(
                            dim=-1, index=input_ids_rmpad_rolled.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             # Use specific IDs (from rollout)
                             topk_ids = student_top_k_ids
                             if student_top_k_ids.ndim == 3: # (bsz, seqlen, k)
                                 # We are in rmpad mode, but student_top_k_ids is padded 3D tensor
                                 # We need to extract the relevant tokens aligning with input_ids_rmpad_rolled
                                 
                                 # This is tricky because student_top_k_ids is shaped (batch, seq, k)
                                 # and logits_rmpad is (total_nnz, vocab)
                                 # We need to flatten student_top_k_ids to (total_nnz, k) using indices
                                 
                                 # Re-use the indices computed from unpad_input
                                 # indices: (total_nnz,) 
                                 # student_top_k_ids: (batch, seq, k)
                                 
                                 # 1. If student_top_k_ids only covers the response, pad it to match full sequence length
                                 if student_top_k_ids.shape[1] != seqlen:
                                     full_student_top_k_ids = torch.zeros((batch_size, seqlen, top_k), 
                                                                         dtype=student_top_k_ids.dtype, 
                                                                         device=student_top_k_ids.device)
                                     full_student_top_k_ids[:, -response_length-1:-1, :] = student_top_k_ids
                                     student_top_k_ids = full_student_top_k_ids

                                 # 2. Flatten student_top_k_ids to (batch*seq, k)
                                 flat_ids = student_top_k_ids.view(-1, top_k)
                                 
                                 # 3. Select using indices
                                 # Note: indices are from attention_mask, which aligns with how logits_rmpad represents data
                                 topk_ids_rmpad = flat_ids[indices] # (total_nnz, k)
                                 
                                 # If 'student_top_k_ids' in batch has shape (batch, seq_len, k), then:
                                 topk_ids = topk_ids_rmpad
                                 
                             else:
                                 # If it's already flattened? Unlikely.
                                 pass

                        else:
                             # Legacy/Resample behavior
                             _, topk_ids = torch.topk(logits_rmpad, k=top_k, dim=-1)

                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if top_k > 0:
                         topk_ids = gather_outputs_and_unpad(
                            topk_ids,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                         topk_log_probs = gather_outputs_and_unpad(
                            topk_log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                         )
                    if return_prior_opd_features and prior_opd_hidden_enable and hidden_available:
                        hidden_rmpad = gather_outputs_and_unpad(
                            hidden_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                
                if top_k > 0:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if return_prior_opd_features and prior_opd_hidden_enable and hidden_available:
                    full_hidden = pad_input(
                        hidden_states=hidden_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    # Hidden state of the response tokens themselves, not their predictor positions.
                    response_hidden = full_hidden[:, -response_length:, :]

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                
                if top_k > 0:
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                if return_prior_opd_features and prior_opd_hidden_enable:
                    extra_args["output_hidden_states"] = True
                    extra_args["return_dict"] = True

                output, boundary_outputs = self._forward_with_boundary_capture(
                    forward_call=lambda: self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    ),
                    settings=boundary_opd_settings,
                    capture_target=boundary_capture_target,
                    capture_resolution_error=boundary_capture_resolution_error,
                    state_indices=boundary_state_indices,
                    transition_valid_mask=boundary_transition_valid_mask,
                    batch_size=batch_size,
                    sequence_length=seqlen,
                    hidden_dim=boundary_hidden_dim,
                    prompt_ids=micro_batch.get("ff_prompt_uid"),
                )  # prevent model thinks we are generating

                if return_prior_opd_features and prior_opd_hidden_enable:
                    hidden_states = getattr(output, "hidden_states", None)
                    if hidden_states:
                        response_hidden = hidden_states[-1][:, -response_length:, :].detach()
                        hidden_available = True
                        output.hidden_states = None
                        del hidden_states
                
                need_logits = top_k > 0
                if self.use_fused_kernels and not need_logits:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    
                    # Optimization: when top_k > 0, compute log_softmax once and gather both
                    # log_probs and topk_log_probs to avoid duplicate computation
                    need_topk = top_k > 0
                    if need_topk:
                        # Compute log_softmax once for both target and topk tokens
                        log_probs_all = torch.log_softmax(logits, dim=-1)
                        # Gather log_probs for target tokens (responses)
                        log_probs = log_probs_all.gather(
                            dim=-1, index=micro_batch["responses"].unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    
                    if need_topk:
                        if student_top_k_ids is not None:
                             topk_ids = student_top_k_ids
                             # Ensure shape alignment if needed, but for non-rmpad (bsz, seq, k) should match logits (bsz, seq, vocab) dim 0,1
                        else:
                             _, topk_ids = torch.topk(logits, k=top_k, dim=-1)
                        
                        # Use pre-computed log_probs_all (always available when need_topk=True)
                        topk_log_probs = log_probs_all.gather(dim=-1, index=topk_ids)

            prior_opd_pooled_hidden = None
            prior_opd_scalar_features = None
            if return_prior_opd_features:
                response_mask = micro_batch.get("response_mask", attention_mask[:, -response_length:])
                prior_opd_pooled_hidden, prior_opd_scalar_features = self._pool_prior_opd_features(
                    response_hidden=response_hidden,
                    response_mask=response_mask,
                    responses=micro_batch["responses"],
                    attention_mask=attention_mask,
                    entropy=entropy,
                    log_probs=log_probs,
                    topk_ids=topk_ids,
                    topk_log_probs=topk_log_probs,
                )

            if return_prior_opd_features:
                forward_outputs = (
                    entropy,
                    log_probs,
                    topk_ids,
                    topk_log_probs,
                    prior_opd_pooled_hidden,
                    prior_opd_scalar_features,
                )
            else:
                forward_outputs = (entropy, log_probs, topk_ids, topk_log_probs)
            if boundary_opd_settings is not None:
                assert boundary_outputs is not None
                return (*forward_outputs, *boundary_outputs)
            return forward_outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_probs_for_ids(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability for specific token ids
        Args:
            data (DataProto): a DataProto containing input_ids, attention_mask, position_ids, responses, 
                             and target_ids (batch, response_len, k) in batch
        Returns:
            torch.Tensor: (batch, response_len, k) log probs for target_ids
        """
        # set to eval
        self.actor_module.eval()

        target_ids = data.batch["target_ids"]
        
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "target_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_log_probs_lst = []
        top_k = target_ids.shape[-1]

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            mb_target_ids = model_inputs["target_ids"]
            with torch.no_grad():
                # We reuse _forward_micro_batch. It returns (entropy, log_probs, topk_ids, topk_log_probs)
                _, _, _, topk_log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=False, 
                    top_k=top_k, student_top_k_ids=mb_target_ids
                )
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # topk_log_probs = topk_log_probs.to("cpu")
            topk_log_probs_lst.append(topk_log_probs)

        topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)

        if use_dynamic_bsz:
            topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)

        return topk_log_probs_tensor

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_distillation_reward(self, data: DataProto) -> DataProto:
        """Compute the distillation reward (rm_scores) on GPU
        Args:
            data (DataProto): containing all necessary tensors for distillation reward calculation
        Returns:
            DataProto: containing rm_scores and other updated tensors (e.g., union_ids)
        """
        # Set to eval mode for forward passes
        self.actor_module.eval()

        # 1. Extract parameters from meta_info
        top_k = data.meta_info.get("selector_top_k", data.meta_info.get("log_prob_top_k", 0))
        strategy = data.meta_info.get(
            "selector_top_k_strategy", data.meta_info.get("top_k_strategy", "only_stu")
        )
        kl_estimator = data.meta_info.get("kl_estimator", "k1")
        reward_weight_mode = data.meta_info.get("reward_weight_mode", "student_p")  # "student_p", "teacher_p", or "none"
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        # 2. Compute Student Log Probs on Teacher IDs if needed
        # (This replaces the previous call to compute_log_probs_for_ids in ray_trainer)
        S_on_T = None
        if strategy in ["only_tch", "intersection", "union", "union-intersection"]:
            target_ids = data.batch["teacher_top_k_ids"]
            
            # Select keys for micro-batching
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            
            # We need to pass target_ids to _forward_micro_batch, but since we are micro-batching, 
            # we should split target_ids as well.
            mb_data = data.select(batch_keys=select_keys + ["teacher_top_k_ids"], 
                                 non_tensor_batch_keys=non_tensor_select_keys)
            
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(mb_data, max_token_len=max_token_len)
            else:
                micro_batches = mb_data.split(micro_batch_size)

            S_on_T_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                mb_target_ids = model_inputs["teacher_top_k_ids"]
                with torch.no_grad():
                    _, _, _, topk_log_probs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=False, 
                        top_k=top_k, student_top_k_ids=mb_target_ids
                    )
                S_on_T_lst.append(topk_log_probs)

            S_on_T = torch.concat(S_on_T_lst, dim=0)
            if use_dynamic_bsz:
                S_on_T = restore_dynamic_batch(S_on_T, batch_idx_list)
        
        # 3. Compute rm_scores on GPU
        # Move all necessary tensors to GPU (they should already be there if passed from fsdp_workers)
        device = get_device_id()
        S_ids = data.batch["student_top_k_ids"].to(device)
        S_logp = data.batch["student_top_k_log_probs"].to(device)
        T_on_S = data.batch["teacher_on_student_log_probs"].to(device)
        
        T_ids = data.batch.get("teacher_top_k_ids", None)
        if T_ids is not None: T_ids = T_ids.to(device)
        T_logp = data.batch.get("teacher_top_k_log_probs", None)
        if T_logp is not None: T_logp = T_logp.to(device)
        overlap_mask = data.batch.get("overlap_mask", None)
        if overlap_mask is not None: overlap_mask = overlap_mask.to(device)

        def compute_reward_weights(S_logp, T_logp, valid_mask, weight_mode, normalize=True):
            """Compute weights for reward calculation.
            
            Args:
                S_logp: Student log probabilities (batch, seq, K)
                T_logp: Teacher log probabilities (batch, seq, K)
                valid_mask: Boolean mask for valid tokens (batch, seq, K)
                weight_mode: "student_p", "teacher_p", or "none"
                normalize: If True, apply softmax normalization across K dim.
                          If False, use raw probabilities (masked by valid_mask).
            
            Returns:
                Weights (batch, seq, K)
            """
            if weight_mode == "student_p":
                log_probs = S_logp
            elif weight_mode == "teacher_p":
                log_probs = T_logp
            elif weight_mode == "none":
                # 对于"none"模式，使用均匀分布
                log_probs = torch.zeros_like(S_logp)
            else:
                raise ValueError(f"Unknown reward_weight_mode: {weight_mode}")
            
            log_probs = torch.where(valid_mask, log_probs, torch.full_like(log_probs, -float('inf')))
            
            if normalize:
                norm_log_weights = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
                weights = torch.exp(norm_log_weights)
            else:
                weights = torch.exp(log_probs)
            
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            
            return weights

        res_tensors = {}
        ta_opd_enable = bool(data.meta_info.get("ta_opd_enable", False))
        
        if strategy == "only_stu":
            kl_val = S_logp - T_on_S
            valid_mask = torch.ones_like(S_logp, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights

        elif strategy == "only_tch":
            kl_val = S_on_T - T_logp
            valid_mask = torch.ones_like(S_on_T, dtype=torch.bool)
            norm_weights = compute_reward_weights(S_on_T, T_logp, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            res_tensors["union_top_k_ids"] = T_ids

        elif strategy == "intersection":
            valid_mask = overlap_mask.bool()
            kl_val = S_logp - T_on_S
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp, T_on_S, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights

        elif strategy == "union":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)
            
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                torch.ones_like(S_ids, dtype=torch.bool),
                ~T_in_S
            ], dim=-1)
            
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode)
            rm_scores = -kl_val * norm_weights
            
            # TA-OPD needs cross-support log-probs for its selector. Other
            # methods may retain union tensors for auxiliary diagnostics.
            if not ta_opd_enable:
                res_tensors["union_top_k_ids"] = union_ids
                res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T

        elif strategy == "union-intersection":
            union_ids = torch.cat([S_ids, T_ids], dim=-1)
            S_logp_union = torch.cat([S_logp, S_on_T], dim=-1)
            T_logp_union = torch.cat([T_on_S, T_logp], dim=-1)

            S_in_T = overlap_mask.bool().to(device)
            T_in_S = data.batch["teacher_in_student_mask"].bool().to(device)
            valid_mask = torch.cat([
                ~S_in_T,    # S_ids is valid if not in T
                ~T_in_S     # T_ids is valid if not in S
            ], dim=-1)
                
            kl_val = S_logp_union - T_logp_union
            kl_val = torch.where(valid_mask, kl_val, torch.zeros_like(kl_val))
            norm_weights = compute_reward_weights(S_logp_union, T_logp_union, valid_mask, reward_weight_mode, normalize=False)
            rm_scores = -kl_val * norm_weights
            
            # Use different keys to avoid conflict with batch's student_top_k_ids
            res_tensors["union_top_k_ids"] = union_ids
            res_tensors["union_top_k_log_probs"] = S_logp_union
            res_tensors["student_log_probs_on_teacher_ids"] = S_on_T

        sampled_opd_rm_scores = data.batch.get("sampled_opd_rm_scores", None)
        if sampled_opd_rm_scores is not None:
            sampled_opd_rm_scores = sampled_opd_rm_scores.to(device)
        
        res_tensors["rm_scores"] = select_opd_training_reward(
            sampled_opd_rm_scores=sampled_opd_rm_scores,
            topk_rm_scores=rm_scores,
            ta_opd_enable=ta_opd_enable,
        )

        return DataProto.from_dict(tensors=res_tensors)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        optimizer_step_applied = bool(torch.isfinite(grad_norm).item())
        if not optimizer_step_applied:
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        self._last_optimizer_step_metrics = {
            "grad_norm_pre_clip": float(grad_norm.detach().item()),
            # FSDP's clip_grad_norm_ API returns the global norm before clipping.
            # It does not expose a measured post-clip global norm.
            "grad_norm_post_clip": None,
            "grad_clip_threshold": float(self.config.grad_clip),
            "grad_clip_applied": int(
                optimizer_step_applied and float(grad_norm.detach().item()) > float(self.config.grad_clip)
            ),
            "optimizer_step_applied": int(optimizer_step_applied),
            "optimizer_step_skipped": int(not optimizer_step_applied),
            "found_inf_or_nan": int(not optimizer_step_applied),
        }
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        return_prior_opd_features = data.meta_info.get("prior_opd_enable", False)
        prior_opd_hidden_enable = data.meta_info.get("prior_opd_hidden_enable", True)
        boundary_opd_enable = data.meta_info.get("boundary_opd_enable", False)
        boundary_opd_settings = None
        boundary_capture_target = None
        boundary_capture_resolution_error = None
        boundary_hidden_dim = None
        if boundary_opd_enable:
            boundary_opd_settings = BoundaryOPDSettings.from_mapping(
                data.meta_info.get("boundary_opd_config")
            )
            try:
                boundary_capture_target = resolve_boundary_capture_target(
                    self.actor_module,
                    capture_method=boundary_opd_settings.capture_method,
                )
            except Exception as error:
                boundary_capture_resolution_error = error
            boundary_hidden_dim = self._boundary_hidden_size(boundary_capture_target)
            self._boundary_capture_metadata = {
                "boundary_hidden_capture_module": (
                    boundary_capture_target.module_path
                    if boundary_capture_target is not None
                    else "unresolved"
                ),
                "boundary_hidden_capture_method": (
                    boundary_capture_target.capture_method
                    if boundary_capture_target is not None
                    else "unresolved"
                ),
                "boundary_hidden_dim": boundary_hidden_dim,
                "boundary_hidden_shape": (
                    boundary_state_count(boundary_opd_settings),
                    boundary_hidden_dim,
                ),
                "boundary_sequence_parallel_size": self.ulysses_sequence_parallel_size,
                "boundary_sequence_parallel_sharded": self.use_ulysses_sp,
                # Sequence shards are gathered before sparse states leave the
                # hook, so the selector always sees full hidden vectors.
                "boundary_sequence_parallel_gathered": True,
                "boundary_tensor_parallel_sharded": False,
            }
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if (return_prior_opd_features or boundary_opd_enable) and "response_mask" in data.batch:
            select_keys.append("response_mask")
        if boundary_opd_enable and "boundary_response_mask" in data.batch:
            select_keys.append("boundary_response_mask")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if boundary_opd_enable and "ff_prompt_uid" in data.non_tensor_batch:
            non_tensor_select_keys.append("ff_prompt_uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        top_k = data.meta_info.get("top_k", 0)
        print(f"In compute_log_prob, top_k: {top_k}")
        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []
        prior_opd_pooled_hidden_lst = []
        prior_opd_scalar_features_lst = []
        boundary_states_lst = []
        boundary_transition_valid_mask_lst = []
        boundary_hidden_capture_success_lst = []
        boundary_hidden_capture_time_ms_lst = []
        boundary_communication_time_ms_lst = []
        boundary_extra_peak_memory_mb_lst = []
        boundary_hidden_capture_error_code_lst = []

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                forward_output = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    top_k=top_k,
                    return_prior_opd_features=return_prior_opd_features,
                    prior_opd_hidden_enable=prior_opd_hidden_enable,
                    boundary_opd_settings=boundary_opd_settings,
                    boundary_capture_target=boundary_capture_target,
                    boundary_capture_resolution_error=boundary_capture_resolution_error,
                    boundary_hidden_dim=boundary_hidden_dim,
                )
                if boundary_opd_enable:
                    boundary_payload = forward_output[-7:]
                    forward_output = forward_output[:-7]
                    (
                        boundary_states,
                        boundary_transition_valid_mask,
                        boundary_hidden_capture_success,
                        boundary_hidden_capture_time_ms,
                        boundary_communication_time_ms,
                        boundary_extra_peak_memory_mb,
                        boundary_hidden_capture_error_code,
                    ) = boundary_payload
                if return_prior_opd_features:
                    entropy, log_probs, topk_ids, topk_log_probs, pooled_hidden, scalar_features = forward_output
                else:
                    entropy, log_probs, topk_ids, topk_log_probs = forward_output
                    pooled_hidden = None
                    scalar_features = None
            # Keep on GPU to avoid expensive CPU-GPU transfer for large top-k
            # log_probs = log_probs.to("cpu")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                # entropy = entropy.to("cpu")
                entropy_lst.append(entropy)
            if top_k > 0:
                # topk_ids = topk_ids.to("cpu")
                # topk_log_probs = topk_log_probs.to("cpu")
                topk_ids_lst.append(topk_ids)
                topk_log_probs_lst.append(topk_log_probs)
            if return_prior_opd_features:
                prior_opd_pooled_hidden_lst.append(pooled_hidden)
                prior_opd_scalar_features_lst.append(scalar_features)
            if boundary_opd_enable:
                boundary_states_lst.append(boundary_states)
                boundary_transition_valid_mask_lst.append(boundary_transition_valid_mask)
                boundary_hidden_capture_success_lst.append(boundary_hidden_capture_success)
                boundary_hidden_capture_time_ms_lst.append(boundary_hidden_capture_time_ms)
                boundary_communication_time_ms_lst.append(boundary_communication_time_ms)
                boundary_extra_peak_memory_mb_lst.append(boundary_extra_peak_memory_mb)
                boundary_hidden_capture_error_code_lst.append(boundary_hidden_capture_error_code)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        
        topk_ids_tensor = None
        topk_log_probs_tensor = None
        if top_k > 0:
            topk_ids_tensor = torch.concat(topk_ids_lst, dim=0)
            topk_log_probs_tensor = torch.concat(topk_log_probs_lst, dim=0)
        prior_opd_pooled_hidden = None
        prior_opd_scalar_features = None
        if return_prior_opd_features:
            prior_opd_pooled_hidden = torch.concat(prior_opd_pooled_hidden_lst, dim=0)
            prior_opd_scalar_features = torch.concat(prior_opd_scalar_features_lst, dim=0)
        if boundary_opd_enable:
            boundary_states = torch.concat(boundary_states_lst, dim=0)
            boundary_transition_valid_mask = torch.concat(boundary_transition_valid_mask_lst, dim=0)
            boundary_hidden_capture_success = torch.concat(boundary_hidden_capture_success_lst, dim=0)
            boundary_hidden_capture_time_ms = torch.concat(boundary_hidden_capture_time_ms_lst, dim=0)
            boundary_communication_time_ms = torch.concat(boundary_communication_time_ms_lst, dim=0)
            boundary_extra_peak_memory_mb = torch.concat(boundary_extra_peak_memory_mb_lst, dim=0)
            boundary_hidden_capture_error_code = torch.concat(
                boundary_hidden_capture_error_code_lst,
                dim=0,
            )

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if top_k > 0:
                topk_ids_tensor = restore_dynamic_batch(topk_ids_tensor, batch_idx_list)
                topk_log_probs_tensor = restore_dynamic_batch(topk_log_probs_tensor, batch_idx_list)
            if return_prior_opd_features:
                prior_opd_pooled_hidden = restore_dynamic_batch(prior_opd_pooled_hidden, batch_idx_list)
                prior_opd_scalar_features = restore_dynamic_batch(prior_opd_scalar_features, batch_idx_list)
            if boundary_opd_enable:
                boundary_states = restore_dynamic_batch(boundary_states, batch_idx_list)
                boundary_transition_valid_mask = restore_dynamic_batch(
                    boundary_transition_valid_mask,
                    batch_idx_list,
                )
                boundary_hidden_capture_success = restore_dynamic_batch(
                    boundary_hidden_capture_success,
                    batch_idx_list,
                )
                boundary_hidden_capture_time_ms = restore_dynamic_batch(
                    boundary_hidden_capture_time_ms,
                    batch_idx_list,
                )
                boundary_communication_time_ms = restore_dynamic_batch(
                    boundary_communication_time_ms,
                    batch_idx_list,
                )
                boundary_extra_peak_memory_mb = restore_dynamic_batch(
                    boundary_extra_peak_memory_mb,
                    batch_idx_list,
                )
                boundary_hidden_capture_error_code = restore_dynamic_batch(
                    boundary_hidden_capture_error_code,
                    batch_idx_list,
                )

        if return_prior_opd_features:
            outputs = (
                log_probs,
                entropys,
                topk_ids_tensor,
                topk_log_probs_tensor,
                prior_opd_pooled_hidden,
                prior_opd_scalar_features,
            )
        else:
            outputs = (log_probs, entropys, topk_ids_tensor, topk_log_probs_tensor)
        if boundary_opd_enable:
            boundary_outputs = (
                *outputs,
                boundary_states,
                boundary_transition_valid_mask,
                boundary_hidden_capture_success,
                boundary_hidden_capture_time_ms,
                boundary_communication_time_ms,
                boundary_extra_peak_memory_mb,
                boundary_hidden_capture_error_code,
            )
            return boundary_outputs
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        ff_ht_enabled = bool(data.meta_info.get("ff_opd_ht_enabled", False))

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if ff_ht_enabled:
            if "ff_opd_query_weight" not in data.batch.keys():
                raise RuntimeError("L8-HT requires ff_opd_query_weight in the Actor batch")
            select_keys.append("ff_opd_query_weight")

        if "format_mask" in data.batch.keys():
            select_keys.append("format_mask") # (bsz, 1)

        # Include TA-OPD selected_mask if present
        if "ta_opd_selected_mask" in data.batch.keys():
            select_keys.append("ta_opd_selected_mask")
        
        # Include student_top_k_log_probs if present (for top-k distillation)
        if "student_top_k_log_probs" in data.batch.keys():
            select_keys.append("student_top_k_log_probs")

        # Include student_top_k_ids if present (for fixing "apples-to-oranges" bug)
        if "student_top_k_ids" in data.batch.keys():
            select_keys.append("student_top_k_ids")

        # Include union_top_k_ids/log_probs for union strategy
        if "union_top_k_ids" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_ids")
            select_keys.append("union_top_k_ids")
            # now we don't need to store student_top_k_ids and student_top_k_log_probs for union strategy
            if "student_top_k_ids" in select_keys:
                select_keys.remove("student_top_k_ids")

        if "union_top_k_log_probs" in data.batch.keys():
            print("Now we are using union strategy, get union_top_k_log_probs")
            select_keys.append("union_top_k_log_probs")
            # now we don't need to store student_top_k_log_probs for union strategy
            if "student_top_k_log_probs" in select_keys:
                select_keys.remove("student_top_k_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        ppo_mini_batch_size = self.config.ppo_mini_batch_size
        if ff_ht_enabled:
            # One optimizer step must cover the whole categorical-sampling
            # window. The count includes zero-mask DP padding rows.
            ppo_mini_batch_size = int(data.meta_info["ff_opd_ht_actor_batch_count"])
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(ppo_mini_batch_size)
        if ff_ht_enabled and len(mini_batches) != 1:
            raise RuntimeError("L8-HT requires exactly one Actor optimizer step per sampling window")

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        opd_audit = {
            "loss_sum": 0.0,
            "token_count": 0.0,
            "opd_loss_raw_sum": 0.0,
            "opd_loss_backward_sum": 0.0,
            "total_loss_raw_sum": 0.0,
            "total_loss_backward_sum": 0.0,
            "backward_count": 0,
            "micro_step_count": 0,
            "optimizer_step_applied": 0,
            "optimizer_step_skipped": 0,
            "grad_norm_pre_clip_sum": 0.0,
            "grad_norm_pre_clip_count": 0,
            "grad_clip_applied": 0,
            "found_inf_or_nan": 0,
        }
        ta_opd_normalize_by = data.meta_info.get("ta_opd_normalize_by", "selected_tokens")
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                ta_opd_selected_token_scale = None
                mini_selected_mask = mini_batch.batch.get("ta_opd_selected_mask", None)
                if mini_selected_mask is not None and ta_opd_normalize_by == "selected_tokens":
                    mini_loss_mask = mini_batch.batch["response_mask"]
                    mini_format_mask = mini_batch.batch.get("format_mask", None)
                    if mini_format_mask is not None:
                        mini_loss_mask = mini_loss_mask * mini_format_mask.unsqueeze(-1)
                    global_selected_count = (mini_selected_mask * mini_loss_mask).sum().float()
                    gradient_average_world_size = 1
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(global_selected_count, op=torch.distributed.ReduceOp.SUM)
                        gradient_average_world_size = torch.distributed.get_world_size()
                    ta_opd_selected_token_scale = torch.where(
                        global_selected_count > 0,
                        global_selected_count.new_tensor(float(gradient_average_world_size)) / global_selected_count,
                        torch.zeros_like(global_selected_count),
                    )

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    opd_audit["micro_step_count"] += 1
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    # Check if we have 3D advantages (top-k sampling case)
                    # If so, we need to recompute top-k log probs for correct gradient
                    if advantages.dim() == 3:
                        top_k = advantages.shape[-1]
                        # For union strategy, use union_top_k_ids; otherwise use student_top_k_ids
                        student_top_k_ids = None
                        if "union_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["union_top_k_ids"]
                        elif "student_top_k_ids" in model_inputs:
                            student_top_k_ids = model_inputs["student_top_k_ids"]

                        entropy, _, _, topk_log_probs = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            top_k=top_k, student_top_k_ids=student_top_k_ids
                        )
                        log_prob_for_loss = topk_log_probs
                        
                    else:
                        _, log_prob, *_ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )
                        log_prob_for_loss = log_prob

                    if ff_ht_enabled:
                        if advantages.dim() == 3:
                            per_token_loss = -(advantages.detach() * log_prob_for_loss).sum(dim=-1)
                        else:
                            per_token_loss = -(advantages.detach() * log_prob_for_loss)
                        ht_numerator = aggregate_ht_opd_numerator(
                            per_token_loss=per_token_loss,
                            response_mask=response_mask,
                            inverse_probability_weight=model_inputs["ff_opd_query_weight"],
                        )
                        normalizer = int(data.meta_info["ff_opd_ht_full_valid_response_tokens"])
                        if normalizer <= 0:
                            raise RuntimeError("L8-HT requires a positive full-candidate token denominator")
                        dp_size = int(data.meta_info["ff_opd_ht_actor_dp_size"])
                        policy_loss = ht_numerator * (dp_size / normalizer)
                        policy_loss.backward()

                        local_token_count = float(response_mask.sum().detach().item())
                        normalized_ht_loss = ht_numerator / normalizer
                        opd_audit["loss_sum"] += float(ht_numerator.detach().item())
                        opd_audit["token_count"] += local_token_count
                        opd_audit["opd_loss_raw_sum"] += float(normalized_ht_loss.detach().item())
                        opd_audit["opd_loss_backward_sum"] += float(policy_loss.detach().item())
                        opd_audit["total_loss_raw_sum"] += float(normalized_ht_loss.detach().item())
                        opd_audit["total_loss_backward_sum"] += float(policy_loss.detach().item())
                        opd_audit["backward_count"] += 1
                        micro_batch_metrics.update(
                            {
                                "actor/pg_loss": normalized_ht_loss.detach().item(),
                                "actor/l8_ht_numerator": ht_numerator.detach().item(),
                                "actor/l8_ht_full_valid_response_tokens": float(normalizer),
                                "actor/l8_ht_actor_dp_size": float(dp_size),
                            }
                        )
                        append_to_dict(metrics, micro_batch_metrics)
                        continue

                    format_mask = None
                    if "format_mask" in model_inputs.keys():
                        format_mask = model_inputs["format_mask"]
            

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            print("on_policy")
                            # For on-policy (ppo_epochs=1), use current policy as "old"
                            # log_prob_for_loss is already 3D for top-k case
                            old_log_prob = log_prob_for_loss.detach()
                        else:
                            print("off_policy")
                            # For off-policy, use stored log probs
                            # For 3D top-k case, use stored log probs (union or student)
                            if advantages.dim() == 3:
                                if "union_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["union_top_k_log_probs"]
                                elif "student_top_k_log_probs" in model_inputs:
                                    old_log_prob = model_inputs["student_top_k_log_probs"]
                                else:
                                    old_log_prob = model_inputs["old_log_probs"]
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Get TA-OPD selected_mask if present
                    ta_opd_selected_mask = model_inputs.get("ta_opd_selected_mask", None)
                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob_for_loss,  # 3D for top-k, 2D otherwise
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        format_mask=format_mask,
                        selected_mask=ta_opd_selected_mask,
                        ta_opd_normalize_by=ta_opd_normalize_by,
                        ta_opd_selected_token_scale=ta_opd_selected_token_scale,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if ta_opd_selected_token_scale is not None:
                        # pg_loss already contains the distributed-global TA
                        # numerator scale. Any optional entropy/KL terms still
                        # need their ordinary micro-batch scaling.
                        loss = pg_loss + (policy_loss - pg_loss) * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    effective_opd_mask = response_mask
                    if format_mask is not None:
                        effective_opd_mask = effective_opd_mask * format_mask.unsqueeze(-1)
                    if ta_opd_selected_mask is not None:
                        effective_opd_mask = effective_opd_mask * ta_opd_selected_mask
                    local_opd_token_count = float(effective_opd_mask.sum().detach().item())
                    local_opd_loss_sum = None
                    if loss_agg_mode == "token-mean":
                        if ta_opd_selected_token_scale is not None:
                            scale_value = float(ta_opd_selected_token_scale.detach().item())
                            local_opd_loss_sum = (
                                float(pg_loss.detach().item()) / scale_value if scale_value > 0 else 0.0
                            )
                        else:
                            local_opd_loss_sum = float(pg_loss.detach().item()) * local_opd_token_count
                    if local_opd_loss_sum is not None:
                        opd_audit["loss_sum"] += local_opd_loss_sum
                    opd_audit["token_count"] += local_opd_token_count
                    opd_audit["opd_loss_raw_sum"] += float(pg_loss.detach().item())
                    opd_backward = pg_loss if ta_opd_selected_token_scale is not None else pg_loss * loss_scale_factor
                    opd_audit["opd_loss_backward_sum"] += float(opd_backward.detach().item())
                    opd_audit["total_loss_raw_sum"] += float(policy_loss.detach().item())
                    opd_audit["total_loss_backward_sum"] += float(loss.detach().item())
                    loss.backward()
                    opd_audit["backward_count"] += 1

                    pg_metric_scale = 1.0 if ta_opd_selected_token_scale is not None else loss_scale_factor
                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * pg_metric_scale
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                step_audit = self._last_optimizer_step_metrics
                opd_audit["optimizer_step_applied"] += step_audit["optimizer_step_applied"]
                opd_audit["optimizer_step_skipped"] += step_audit["optimizer_step_skipped"]
                opd_audit["grad_clip_applied"] += step_audit["grad_clip_applied"]
                opd_audit["found_inf_or_nan"] += step_audit["found_inf_or_nan"]
                if math.isfinite(step_audit["grad_norm_pre_clip"]):
                    opd_audit["grad_norm_pre_clip_sum"] += step_audit["grad_norm_pre_clip"]
                    opd_audit["grad_norm_pre_clip_count"] += 1
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        audit_divisor = max(opd_audit["backward_count"], 1)
        grad_divisor = max(opd_audit["grad_norm_pre_clip_count"], 1)
        metrics.update(
            {
                "opd_audit/loss_sum": [opd_audit["loss_sum"]],
                "opd_audit/token_count": [opd_audit["token_count"]],
                "opd_audit/opd_loss_raw": [opd_audit["opd_loss_raw_sum"] / audit_divisor],
                "opd_audit/opd_loss_backward": [opd_audit["opd_loss_backward_sum"] / audit_divisor],
                "opd_audit/total_loss_raw": [opd_audit["total_loss_raw_sum"] / audit_divisor],
                "opd_audit/total_loss_backward": [opd_audit["total_loss_backward_sum"] / audit_divisor],
                "opd_audit/backward_executed": [int(opd_audit["backward_count"] > 0)],
                "opd_audit/micro_step": [opd_audit["micro_step_count"]],
                "opd_audit/gradient_accumulation_step": [int(getattr(self, "gradient_accumulation", 1))],
                "opd_audit/optimizer_step_applied": [opd_audit["optimizer_step_applied"]],
                "opd_audit/optimizer_step_skipped": [opd_audit["optimizer_step_skipped"]],
                "opd_audit/grad_norm_pre_clip": [opd_audit["grad_norm_pre_clip_sum"] / grad_divisor],
                "opd_audit/grad_clip_threshold": [float(self.config.grad_clip)],
                "opd_audit/grad_clip_applied": [int(opd_audit["grad_clip_applied"] > 0)],
                "opd_audit/found_inf_or_nan": [int(opd_audit["found_inf_or_nan"] > 0)],
                "opd_audit/zero_valid_opd_tokens": [int(opd_audit["token_count"] == 0)],
                "opd_audit/opd_loss_weight": [1.0],
                "opd_audit/amp_loss_scale": [1.0],
                "opd_audit/data_parallel_world_size": [
                    torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                ],
            }
        )
        self.actor_optimizer.zero_grad()
        return metrics
