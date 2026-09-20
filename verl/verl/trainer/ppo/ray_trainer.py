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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import csv
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
from omegaconf import OmegaConf, open_dict
import ray
import torch
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

_logger = logging.getLogger(__name__)

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.utils.boundary_opd import (
    BoundaryOPDSettings,
    boundary_transition_count,
    is_boundary_selector_mode,
    stable_group_sibling_indices,
)
from verl.utils.boundary_calibration import BoundaryCalibrationAccumulator
from verl.utils.cost_meter import CostMeter
from verl.utils.ff_opd import (
    TARGET_MODE as FF_TARGET_MODE,
    BoundaryContrastCSVWriter,
    FFCSVWriter,
    FFOPDConfig,
    FFOPDQueueManager,
    FFProfileJSONLWriter,
    make_prompt_uid,
    sampled_reverse_kl_statistics,
    teacher_dummy_padding_size,
)
from verl.utils.ta_opd import compute_opd_metrics, compute_ta_opd_mask
from verl.utils.tlr_opd import (
    TLR_MODE,
    TLRCSVWriter,
    TLRConfig,
    select_tlr_trajectories,
    validate_method_exclusivity,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.csv_utils import build_metric_csv_rows
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_gradient_diagnostics import ROLLOUT_DIAGNOSTIC_KEYS, summarize_rollout_gradient_diagnostics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


@dataclass
class TeacherFilterStats:
    """Counts from the Teacher pre-forward filtering step.

    Attributes:
        n_before:              Total candidates entering the filter.
        drop_overlength:       Dropped because prompt+response exceeds Teacher max length.
        drop_truncated:        Dropped because response has no EOS token (generation cut off).
        drop_invalid:          Dropped due to corrupted tensor / token data.
        n_real:                Valid candidates after all drops (= n_before - all drops).
        n_dummy:               Dummy samples added for DP alignment.
        n_padded:              Total batch size sent to Teacher (= n_real + n_dummy).
        teacher_dp:            Teacher data-parallel world size used for alignment.
    """

    n_before: int = 0
    drop_overlength: int = 0
    drop_truncated: int = 0
    drop_invalid: int = 0
    n_real: int = 0
    n_dummy: int = 0
    n_padded: int = 0
    teacher_dp: int = 1


def filter_truncated_rollouts(
    batch: DataProto,
    eos_token_id,
    dp_size: int = 1,
    allow_truncated: bool = False,
) -> tuple[DataProto, TeacherFilterStats]:
    """Filter truly invalid rollouts and return stats; do NOT drop for DP alignment.

    By default, removes rollouts whose response contains no EOS token. TLR
    passes ``allow_truncated=True`` because the published BoN candidate set
    includes max-length rollouts.

    DP alignment is handled separately by :func:`pad_teacher_batch_with_dummy`,
    which adds dummy samples rather than discarding valid candidates.

    Args:
        batch:        The batch to filter.
        eos_token_id: The EOS token id(s) used by the tokenizer (int or list).
        dp_size:      Teacher data-parallel world size (used only for stats).
        allow_truncated: Keep no-EOS/max-length rollouts for TLR.

    Returns:
        Tuple of (filtered DataProto, TeacherFilterStats).  The returned batch
        contains only real, valid rollouts; its length may not be divisible by
        dp_size yet.
    """
    n_before = len(batch)
    responses = batch.batch["responses"]
    eos_tensor = torch.as_tensor(
        eos_token_id if isinstance(eos_token_id, list) else [eos_token_id],
        device=responses.device,
    )
    has_eos = torch.isin(responses, eos_tensor).any(dim=-1)  # (B,)
    drop_truncated = 0 if allow_truncated else int((~has_eos).sum().item())
    if drop_truncated > 0:
        batch = batch.select_idxs(has_eos.cpu().numpy())

    # NOTE: DP alignment is intentionally NOT done here.  Valid candidates must
    # never be discarded to satisfy a divisibility constraint.  Call
    # pad_teacher_batch_with_dummy() after this function to add dummy samples.
    n_real = len(batch)
    stats = TeacherFilterStats(
        n_before=n_before,
        drop_overlength=0,
        drop_truncated=drop_truncated,
        drop_invalid=0,
        n_real=n_real,
        n_dummy=0,   # filled in by pad_teacher_batch_with_dummy
        n_padded=n_real,  # updated after padding
        teacher_dp=dp_size,
    )
    return batch, stats


def pad_teacher_batch_with_dummy(
    batch: DataProto,
    dp_size: int,
    stats: TeacherFilterStats,
) -> tuple[DataProto, TeacherFilterStats]:
    """Pad *batch* to the next multiple of *dp_size* using dummy samples.

    Dummy samples are copies of the first real sample with all response /
    loss masks zeroed out and an ``is_dummy`` flag set to ``True``.  They are
    used only to satisfy the Teacher DP shape requirement and must be removed
    from all downstream computations (loss, statistics, selector updates).

    NOTE: ``attention_mask`` is intentionally kept intact.  The Teacher
    forward uses remove-padding, and a fully-masked sample would be packed
    into an empty micro-batch (sequence length 0), crashing the model
    forward with an ambiguous reshape error.  Keeping the mask also costs
    nothing: the dummy is excluded from every downstream computation via
    ``response_mask == 0`` and :func:`strip_dummy_samples`.

    The function updates *stats* in-place and also returns it for convenience.

    Args:
        batch:   Real-only DataProto (output of :func:`filter_truncated_rollouts`).
        dp_size: Teacher data-parallel world size.
        stats:   TeacherFilterStats from the filter step (mutated in-place).

    Returns:
        Tuple of (padded DataProto, updated TeacherFilterStats).
    """
    n_real = len(batch)
    if n_real == 0 or dp_size <= 1:
        stats.n_dummy = 0
        stats.n_padded = n_real
        return batch, stats

    n_dummy = teacher_dummy_padding_size(n_real, dp_size)
    if n_dummy == 0:
        stats.n_dummy = 0
        stats.n_padded = n_real
        return batch, stats

    # Build a single dummy template from the first real sample.
    template = batch[:1]  # DataProto of length 1

    # Determine device from the first available tensor in the batch.
    _first_tensor = next(iter(template.batch.values()))
    _device = _first_tensor.device

    # Zero out the response / loss masks so the dummy contributes nothing to
    # any loss or statistic.  ``attention_mask`` must NOT be zeroed: under the
    # Teacher's remove-padding forward it would produce an empty sequence and
    # crash the model (cannot reshape tensor of 0 elements).  Zero contribution
    # is already guaranteed by response_mask/loss_mask == 0 and by
    # strip_dummy_samples() after the Teacher call.
    dummy_batch_td = template.batch.clone()
    for key in list(dummy_batch_td.keys()):
        if key in ("response_mask", "loss_mask"):
            dummy_batch_td[key] = torch.zeros_like(dummy_batch_td[key])
    # Mark as dummy in the tensor batch (float 0/1 flag).
    dummy_batch_td["is_dummy"] = torch.ones(1, dtype=torch.float32, device=_device)

    # Ensure real samples have is_dummy = 0.
    real_batch_td = batch.batch.clone()
    real_batch_td["is_dummy"] = torch.zeros(n_real, dtype=torch.float32, device=_device)

    # Replicate the dummy template n_dummy times.
    dummy_protos = []
    for _ in range(n_dummy):
        dummy_non_tensor = {k: v[:1].copy() for k, v in template.non_tensor_batch.items()}
        dummy_proto = DataProto(
            batch=dummy_batch_td.clone(),
            non_tensor_batch=dummy_non_tensor,
            meta_info={},
        )
        dummy_protos.append(dummy_proto)

    # Rebuild real DataProto with is_dummy=0 field.
    real_proto = DataProto(
        batch=real_batch_td,
        non_tensor_batch=batch.non_tensor_batch,
        meta_info=batch.meta_info,
    )

    padded = DataProto.concat([real_proto] + dummy_protos)

    stats.n_dummy = n_dummy
    stats.n_padded = n_real + n_dummy
    return padded, stats


def strip_dummy_samples(batch: DataProto, n_real: int) -> DataProto:
    """Remove dummy samples appended by :func:`pad_teacher_batch_with_dummy`.

    Args:
        batch:  Padded DataProto (real + dummy samples).
        n_real: Number of real samples at the front of the batch.

    Returns:
        DataProto containing only the first *n_real* rows.
    """
    if n_real >= len(batch):
        return batch
    return batch.slice(end=n_real)


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "true_reward_score" in data.batch: # optional
            adv_kwargs["true_reward_score"] = data.batch["true_reward_score"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        res = adv_estimator_fn(**adv_kwargs)
        if len(res) == 2:
            advantages, returns = res
        elif len(res) == 3:
            advantages, returns, extra_metrics = res
            for k, v in extra_metrics.items():
                data.batch[k] = v
        else:
            raise ValueError("Invalid return from adv_estimator_fn")

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        rollout_config = self.config.actor_rollout_ref.rollout
        self._ff_opd_enabled = bool(self.config.get("ff_opd", {}).get("enable", False))
        self._ff_opd_manager = None
        self._ff_opd_csv = None
        self._ff_opd_profiles = None
        self._ff_kl_profiles = None
        self._boundary_contrast_csv = None
        self._ff_prompt_store = {}
        self._boundary_opd_enabled = False
        self._boundary_hidden_capture_enabled = False
        self._boundary_calibration = None
        self._boundary_opd_worker_config = None
        self._tlr_opd_enabled = bool(self.config.get("tlr_opd", {}).get("enabled", False))
        self._tlr_config = None
        self._tlr_csv = None
        if self._tlr_opd_enabled:
            self._tlr_config = TLRConfig.from_mapping(self.config.tlr_opd)
            self._tlr_config.validate(rollout_n=int(rollout_config.n))
            conflicting = {
                "ff_opd": self._ff_opd_enabled,
                "gq_opd": bool(
                    self.config.get("gq_opd", {}).get(
                        "enabled", self.config.get("gq_opd", {}).get("enable", False)
                    )
                ),
                "random_query_opd": bool(
                    self.config.get("random_query_opd", {}).get(
                        "enabled",
                        self.config.get("random_query_opd", {}).get("enable", False),
                    )
                ),
            }
            validate_method_exclusivity(True, **conflicting)
            if bool(rollout_config.get("ta_opd_enable", False)):
                raise ValueError("TLR-OPD cannot be combined with post-query TA-OPD")
            if not self.use_rm:
                raise ValueError("TLR-OPD requires an enabled Teacher reward model")
            if str(self.config.algorithm.adv_estimator) != "token_reward_direct":
                raise ValueError("TLR-OPD requires algorithm.adv_estimator=token_reward_direct")
            if self.config.actor_rollout_ref.actor.loss_agg_mode != "seq-mean-token-mean":
                raise ValueError(
                    "TLR-OPD requires actor loss_agg_mode=seq-mean-token-mean "
                    "for the original per-trajectory OPSD objective"
                )
            self._tlr_csv = TLRCSVWriter(
                self._tlr_config.csv_path if self._tlr_config.save_csv else ""
            )
        if self._ff_opd_enabled:
            ff_config = FFOPDConfig.from_mapping(self.config.ff_opd)
            ff_config.selector_mode = str(self.config.algorithm.ff_selector_mode)
            ff_config.seed = int(self.config.trainer.seed)
            self._boundary_opd_enabled = is_boundary_selector_mode(
                ff_config.selector_mode
            )
            if not self._boundary_opd_enabled:
                # ff_opd.* carries no selector mode, so FFOPDConfig.__post_init__
                # resolves the dataclass default (boundary_opd) and attaches
                # placeholder Boundary settings. The authoritative mode arrives
                # from algorithm.ff_selector_mode above; drop the placeholder so
                # the legacy selectors keep algorithm.boundary_opd unused.
                ff_config.boundary_opd = None
            if self._boundary_opd_enabled:
                boundary_config = OmegaConf.select(
                    self.config, "algorithm.boundary_opd"
                )
                ff_config.boundary_opd = BoundaryOPDSettings.from_mapping(
                    boundary_config
                )
                # The Boundary selector always captures one sparse hidden state
                # per response boundary from the Student's log-prob forward.
                self._boundary_hidden_capture_enabled = True
                self._boundary_opd_worker_config = OmegaConf.to_container(
                    boundary_config, resolve=True
                )
                # Sparse pre-LM-head capture is implemented in the FSDP data
                # parallel actor only. Fail loudly instead of degrading every
                # prompt into the FF-Cost fallback on another backend.
                actor_strategy = str(self.config.actor_rollout_ref.actor.strategy)
                if actor_strategy not in {"fsdp", "fsdp2"}:
                    raise ValueError(
                        "Boundary-OPD hidden capture requires "
                        "actor_rollout_ref.actor.strategy in {fsdp, fsdp2}, got "
                        f"{actor_strategy!r}"
                    )
            ff_config.validate(
                rollout_n=rollout_config.n,
                opd_target_mode=str(rollout_config.get("opd_target_mode", FF_TARGET_MODE)),
            )
            if not self.use_rm:
                raise ValueError("FF-OPD requires an enabled Teacher reward model")
            if rollout_config.get("ta_opd_enable", False):
                raise ValueError("FF-OPD cannot be combined with TA-OPD")
            if str(self.config.algorithm.adv_estimator) != "token_reward_direct":
                raise ValueError("FF-OPD requires algorithm.adv_estimator=token_reward_direct")
            if self.config.actor_rollout_ref.actor.loss_agg_mode != "token-mean":
                raise ValueError("FF-OPD requires the existing Full OPD token-mean reduction")
            if self.config.actor_rollout_ref.actor.ppo_epochs != 1:
                raise ValueError("FF-OPD requires actor_rollout_ref.actor.ppo_epochs=1")
            if self.config.actor_rollout_ref.actor.entropy_coeff != 0:
                raise ValueError("FF-OPD requires actor entropy_coeff=0")
            if self.config.actor_rollout_ref.actor.use_kl_loss:
                raise ValueError("FF-OPD requires actor use_kl_loss=False")
            if self.config.trainer.critic_warmup != 0:
                raise ValueError("FF-OPD requires trainer.critic_warmup=0")
            if ff_config.base_epochs != 1:
                raise ValueError("FF-OPD requires ff_opd.base_epochs=1")
            if int(self.config.trainer.total_epochs) < 1:
                raise ValueError(
                    "FF-OPD requires trainer.total_epochs >= 1; each dataset epoch re-runs the "
                    "fresh pass with the current student and saves its own checkpoint"
                )
            run_name = str(self.config.trainer.experiment_name)
            self._ff_opd_manager = FFOPDQueueManager(
                ff_config, run_name=run_name, tokenizer=self.tokenizer
            )
            if self._boundary_opd_enabled and ff_config.boundary_opd.calibration_collect:
                self._boundary_calibration = BoundaryCalibrationAccumulator(
                    ff_config.boundary_opd,
                    k_rollouts=ff_config.k_rollouts,
                    seed=ff_config.seed,
                )
            self._ff_opd_csv = FFCSVWriter(ff_config.csv_path, ff_config.csv_flush_interval)
            # Full confidence profiles stay out of ff.csv; only sampled audit
            # rows are mirrored into the JSONL sidecar.
            self._ff_opd_profiles = FFProfileJSONLWriter(
                ff_config.profile_jsonl_path, ff_config.csv_flush_interval
            )
            if ff_config.log_kl_profiles:
                # Post-Teacher per-token KL sink for the selected samples.
                self._ff_kl_profiles = FFProfileJSONLWriter(
                    ff_config.kl_profile_jsonl_path, ff_config.csv_flush_interval
                )
            if self._boundary_opd_enabled:
                # Per-candidate Boundary-Contrast profiles (one row per Frontier
                # negative sibling) live next to ff.csv.
                boundary_contrast_csv_path = str(Path(ff_config.csv_path).with_name("boundary_contrast.csv"))
                self._boundary_contrast_csv = BoundaryContrastCSVWriter(
                    boundary_contrast_csv_path,
                    boundary_transition_count(ff_config.boundary_opd),
                )
            full_fresh_total_steps = len(self.train_dataloader)
            self.fresh_total_steps = (
                min(full_fresh_total_steps, ff_config.fresh_step_limit)
                if ff_config.fresh_step_limit > 0
                else full_fresh_total_steps
            )
            self._ff_fresh_step_limited = self.fresh_total_steps < full_fresh_total_steps
            self._ff_fresh_step = 0
            self._ff_retry_step = 0
            self._ff_optimizer_step = 0
            # fresh_total_steps stays per-epoch: it gates the end-of-epoch
            # checkpoint and the fresh progress bar, and is reset each epoch.
            # The LR schedule, however, spans every dataset epoch's fresh pass,
            # so its horizon is fresh_total_steps * total_epochs. Retry updates
            # freeze the scheduler at the LR reached after this horizon.
            ff_total_epochs = max(1, int(self.config.trainer.total_epochs))
            self.total_training_steps = self.fresh_total_steps * ff_total_epochs
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps









    def _append_opd_metrics_csv(self, filename: str, fieldnames: list[str], rows: list[dict]) -> None:
        metrics_dir = self.config.actor_rollout_ref.rollout.get("opd_metrics_dir", None)
        if not metrics_dir or not rows:
            return
        os.makedirs(metrics_dir, exist_ok=True)
        path = os.path.join(metrics_dir, filename)
        if filename == "step_metrics.csv" and os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as stream:
                completed_steps = {
                    int(row.get("global_step", row.get("step", -1))) for row in csv.DictReader(stream)
                }
            rows = [
                row for row in rows
                if int(row.get("global_step", row.get("step", -1))) not in completed_steps
            ]
            if not rows:
                return
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        if not write_header:
            with open(path, newline="", encoding="utf-8") as stream:
                existing_fields = next(csv.reader(stream), [])
            if existing_fields != fieldnames:
                raise RuntimeError(
                    f"{path} has an incompatible CSV header. "
                    "Start a fresh metrics directory before changing OPD metric schemas; "
                    f"existing={existing_fields}, expected={fieldnames}"
                )
        with open(path, "a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _batch_identity(batch: DataProto, key: str, index: int, default):
        values = batch.non_tensor_batch.get(key, None)
        if values is None:
            return default
        value = values[index]
        return value.item() if hasattr(value, "item") else value

    def _write_opd_step_metrics(self, metrics: dict, n_gpus: int) -> None:
        if not self.config.actor_rollout_ref.rollout.get("opd_metrics_dir", None):
            return
        lookup = lambda key, default="": metrics.get(key, default)
        paper_rows = build_metric_csv_rows(metrics, self.global_steps)
        self._append_opd_metrics_csv("paper_metrics.csv", ["step", "metric", "value"], paper_rows)
        validation_rows = build_metric_csv_rows(
            metrics, self.global_steps, prefixes=("val-core/", "val-aux/")
        )
        self._append_opd_metrics_csv(
            "validation_metrics.csv", ["step", "metric", "value"], validation_rows
        )
        step_time = float(lookup("perf/time_per_step", lookup("timing_s/step", 0.0)) or 0.0)
        generation_time = float(lookup("timing_s/gen", 0.0) or 0.0)
        teacher_time = float(lookup("cost/teacher_latency_sec", 0.0) or 0.0)
        actor_time = float(lookup("timing_s/update_actor", 0.0) or 0.0)
        teacher_gpu_count = int(getattr(self.rm_wg, "world_size", n_gpus)) if self.use_rm else 0
        audit_world_size = int(lookup("opd_audit/data_parallel_world_size", n_gpus) or n_gpus)
        global_opd_loss_sum = float(lookup("opd_audit/loss_sum", 0.0) or 0.0) * audit_world_size
        global_opd_token_count = float(lookup("opd_audit/token_count", 0.0) or 0.0) * audit_world_size
        row = {
            "step": self.global_steps,
            "global_step": self.global_steps,
            "optimizer_step": int(lookup("ff/optimizer_step", self.global_steps) or self.global_steps),
            "micro_step": lookup("opd_audit/micro_step", 0),
            "gradient_accumulation_step": lookup("opd_audit/gradient_accumulation_step", 1),
            "data_parallel_world_size": audit_world_size,
            "opd_loss_raw": (
                global_opd_loss_sum / global_opd_token_count
                if global_opd_token_count > 0
                and self.config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
                else lookup("opd_audit/opd_loss_raw")
            ),
            "opd_loss_backward": lookup("opd_audit/opd_loss_backward"),
            "opd_loss_sum": global_opd_loss_sum,
            "opd_loss_token_count": global_opd_token_count,
            "opd_loss_weight": lookup("opd_audit/opd_loss_weight", 1.0),
            "opd_loss_reduction": self.config.actor_rollout_ref.actor.loss_agg_mode,
            "total_loss_raw": lookup("opd_audit/total_loss_raw"),
            "total_loss_backward": lookup("opd_audit/total_loss_backward"),
            "backward_executed": int(bool(lookup("opd_audit/backward_executed", 0))),
            "optimizer_step_applied": int(float(lookup("opd_audit/optimizer_step_applied", 0) or 0) > 0),
            "optimizer_step_skipped": int(float(lookup("opd_audit/optimizer_step_skipped", 0) or 0) > 0),
            "grad_norm_pre_clip": lookup("opd_audit/grad_norm_pre_clip"),
            "grad_norm_post_clip": "",
            "grad_norm_post_clip_available": 0,
            "grad_clip_threshold": lookup("opd_audit/grad_clip_threshold"),
            "grad_clip_applied": int(bool(lookup("opd_audit/grad_clip_applied", 0))),
            "learning_rate": lookup("actor/lr"),
            "amp_loss_scale": lookup("opd_audit/amp_loss_scale", 1.0),
            "found_inf_or_nan": int(bool(lookup("opd_audit/found_inf_or_nan", 0))),
            "zero_valid_opd_tokens": int(global_opd_token_count == 0),
            "teacher_queried_rollouts": (
                lookup("tlr/teacher_real_rollouts", 0)
                if self._tlr_opd_enabled
                else lookup("ff/teacher_queried", 0)
            ),
            "student_generated_tokens": lookup("cost/student_generated_response_token_count"),
            "teacher_input_tokens": lookup("cost/teacher_processed_input_token_count"),
            "teacher_scored_tokens": lookup("cost/teacher_scored_response_token_count"),
            "final_supervised_tokens": lookup("cost/final_supervised_token_count"),
            "student_generation_time_s": generation_time,
            "teacher_forward_time_s": teacher_time,
            "actor_update_time_s": actor_time,
            "step_time_s": step_time,
            "teacher_gpu_hours": teacher_gpu_count * teacher_time / 3600.0,
            "student_gpu_hours": max(
                0.0,
                n_gpus * step_time - teacher_gpu_count * teacher_time,
            ) / 3600.0,
            "e2e_gpu_hours": n_gpus * step_time / 3600.0,
            "communication_time_s": "",
            "other_time_s": max(
                0.0, step_time - generation_time - teacher_time - actor_time
            ),
        }
        self._append_opd_metrics_csv("step_metrics.csv", list(row), [row])
        self._write_ff_ablation_step_metrics(metrics)

    def _write_ff_ablation_step_metrics(self, metrics: dict) -> None:
        """Append one rank-0 FF ablation row, without duplicating resumed steps."""
        if not self._ff_opd_enabled:
            return
        path = Path(self._ff_opd_manager.config.csv_path).with_name(
            "ff_ablation_step_metrics.csv"
        )
        fields = [
            "run_name",
            "selector_mode",
            "seed",
            "global_step",
            "fresh_step",
            "retry_step",
            "optimizer_step",
            "optimizer_updates_this_step",
            "fresh_prompt_attempts",
            "retry_prompt_attempts",
            "generated_rollouts",
            "realized_query_rate",
            "realized_query_rate_cumulative",
            "frontier_prompt_count",
            "target_query_count",
            "actual_query_count",
            "selected_valid_count",
            "selected_correct_count",
            "selected_wrong_count",
            "selected_frontier_count",
            "selected_nonfrontier_count",
            "selected_distance_mean",
            "selected_cost_mean",
            "selected_objective_mean",
            "teacher_input_tokens",
            "teacher_scored_tokens",
            "teacher_latency_sec",
            "time_per_step_sec",
            "boundary/selected_distance_mean",
            "boundary/selected_cost_mean",
            "boundary/selected_objective_mean",
            "boundary/cost_switch_rate",
            "boundary/confidence_distance_spearman",
            "boundary/fallback_rate",
            "boundary/valid_transition_ratio",
            "boundary/hidden_capture_time_ms",
            "boundary/transition_build_time_ms",
            "boundary/distance_compute_time_ms",
            "boundary/communication_time_ms",
            "boundary/extra_peak_memory_mb",
            "boundary/alpha_star_mean",
            "boundary/alpha_star_median",
            "boundary/alpha_star_p25",
            "boundary/alpha_star_p75",
        ]
        selector_mode = self._ff_opd_manager.config.selector_mode
        if selector_mode == "shortest_wrong":
            fields.extend(
                [
                    "shortest_wrong/eligible_prompt_count",
                    "shortest_wrong/selected_count",
                    "shortest_wrong/composition_4p0n",
                    "shortest_wrong/composition_3p1n",
                    "shortest_wrong/composition_2p2n",
                    "shortest_wrong/composition_1p3n",
                    "shortest_wrong/composition_0p4n",
                    "shortest_wrong/selected_length_mean",
                    "shortest_wrong/unselected_wrong_length_mean",
                    "shortest_wrong/min_wrong_length_mean",
                    "shortest_wrong/max_wrong_length_mean",
                    "shortest_wrong/selected_is_global_shortest_fraction",
                    "shortest_wrong/length_margin_mean",
                ]
            )
        elif selector_mode == "frontier_tlr":
            fields.extend(
                [
                    "frontier_tlr/eligible_prompt_count",
                    "frontier_tlr/selected_count",
                    "frontier_tlr/composition_4p0n",
                    "frontier_tlr/composition_3p1n",
                    "frontier_tlr/composition_2p2n",
                    "frontier_tlr/composition_1p3n",
                    "frontier_tlr/composition_0p4n",
                    "frontier_tlr/selected_length_mean",
                    "frontier_tlr/unselected_wrong_length_mean",
                    "frontier_tlr/selected_entropy_mean",
                    "frontier_tlr/unselected_wrong_entropy_mean",
                    "frontier_tlr/selected_score_lh_mean",
                ]
            )
        row = {
            "run_name": str(self.config.trainer.experiment_name),
            "selector_mode": selector_mode,
            "seed": int(self.config.trainer.seed),
            "global_step": int(self.global_steps),
            "fresh_step": metrics.get("ff/fresh_step", 0.0),
            "retry_step": metrics.get("ff/retry_step", 0.0),
            "optimizer_step": metrics.get("ff/optimizer_step", 0.0),
            "optimizer_updates_this_step": metrics.get("ff/actual_optimizer_steps", 0.0),
            "fresh_prompt_attempts": metrics.get("ff/fresh_prompt_attempts", 0.0),
            "retry_prompt_attempts": metrics.get("ff/retry_prompt_attempts", 0.0),
            "generated_rollouts": metrics.get("ff/generated_rollouts", 0.0),
            "realized_query_rate": metrics.get("ff/realized_query_rate", 0.0),
            "realized_query_rate_cumulative": metrics.get(
                "ff/realized_query_rate_cumulative", 0.0
            ),
            **{
                field: metrics.get(f"ff_ablation/{field}", 0.0)
                for field in fields
                if field.startswith("frontier_")
                or field.startswith("target_")
                or field.startswith("actual_")
                or field.startswith("selected_")
            },
            "teacher_input_tokens": metrics.get(
                "cost/teacher_processed_input_token_count", 0.0
            ),
            "teacher_scored_tokens": metrics.get(
                "cost/teacher_scored_response_token_count", 0.0
            ),
            "teacher_latency_sec": metrics.get("cost/teacher_latency_sec", 0.0),
            "time_per_step_sec": metrics.get(
                "perf/time_per_step", metrics.get("timing_s/step", 0.0)
            ),
            **{
                field: metrics.get(field, 0.0)
                for field in fields
                if field.startswith(("boundary/", "shortest_wrong/", "frontier_tlr/"))
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        if not write_header:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != fields:
                    raise RuntimeError(
                        f"{path} has incompatible columns: {reader.fieldnames}"
                    )
                key = (row["run_name"], str(row["global_step"]))
                if any(
                    (existing.get("run_name"), existing.get("global_step")) == key
                    for existing in reader
                ):
                    return
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())





    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None and self.config.data.get("val_files", None):
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=not bool(self.config.get("ff_opd", {}).get("enable", False)),
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataloader = None
        if self.val_dataset is not None:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        if self.val_dataloader is not None:
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader) if self.val_dataloader is not None else 0}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        if self._ff_opd_enabled and self._ff_opd_manager.config.save_queue_state:
            self._ff_opd_csv.flush()
            self._ff_opd_profiles.flush()
            if self._ff_kl_profiles is not None:
                self._ff_kl_profiles.flush()
            torch.save(
                {
                    "schema_version": 4,
                    "manager": self._ff_opd_manager.state_dict(),
                    "prompt_store": self._ff_prompt_store,
                    "written_csv_keys": sorted(self._ff_opd_csv.written_keys),
                    "written_profile_keys": sorted(self._ff_opd_profiles.written_keys),
                    "fresh_step": self._ff_fresh_step,
                    "retry_step": self._ff_retry_step,
                    "optimizer_step": self._ff_optimizer_step,
                },
                os.path.join(local_global_step_folder, "ff_opd.pt"),
            )

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        if self._ff_opd_enabled and self._ff_opd_manager.config.save_queue_state:
            ff_path = os.path.join(global_step_folder, "ff_opd.pt")
            if os.path.exists(ff_path):
                state = torch.load(ff_path, weights_only=False)
                self._ff_opd_manager.load_state_dict(state["manager"])
                self._ff_prompt_store = state.get("prompt_store", {})
                self._ff_opd_csv.written_keys.update(state.get("written_csv_keys", []))
                self._ff_opd_profiles.written_keys.update(state.get("written_profile_keys", []))
                self._ff_fresh_step = int(state.get("fresh_step", 0))
                self._ff_retry_step = int(state.get("retry_step", 0))
                self._ff_optimizer_step = int(state.get("optimizer_step", 0))
                print("Restored FF-OPD queues, counters, prompt store, query accounting, and CSV keys")
            else:
                print("Warning: FF-OPD state missing; starting with fresh queues and budget")

    def _select_tlr_queries(self, batch: DataProto) -> tuple[DataProto | None, dict]:
        """Route the best paper-TLR-scored sibling (short + low entropy) before the Teacher forward."""
        config = self._tlr_config
        n_rollouts = len(batch)
        response_mask = batch.batch["response_mask"].bool()
        finite_logp = torch.isfinite(batch.batch["old_log_probs"]).logical_or(~response_mask).all(dim=-1)
        valid_counts = response_mask.sum(dim=-1).detach().cpu()
        # Ordinary TLR is correctness-agnostic. Frontier-TLR owns the separate
        # mixed-only/wrong-only baseline. The min_rollout_tokens floor is
        # applied INSIDE the selector and BEFORE normalization, so immediate-
        # EOS degenerates can never corrupt the group min/max normalizers.
        eligible = response_mask.any(dim=-1) & finite_logp

        raw_prompt_ids = batch.non_tensor_batch.get("uid")
        if raw_prompt_ids is None:
            raise RuntimeError("TLR requires the rollout group id in non_tensor_batch['uid']")
        prompt_ids = [str(uid) for uid in raw_prompt_ids]
        unique_prompt_ids = list(dict.fromkeys(prompt_ids))
        counts = {uid: prompt_ids.count(uid) for uid in unique_prompt_ids}
        bad_groups = {uid: count for uid, count in counts.items() if count != config.rollouts_per_prompt}
        if bad_groups:
            raise AssertionError(
                "TLR sibling grouping mismatch before routing; expected "
                f"{config.rollouts_per_prompt} rollouts per prompt, got {bad_groups}"
            )

        denom = valid_counts.clamp_min(1)
        entropies = batch.batch.get("entropys")
        if entropies is None:
            raise RuntimeError(
                "Selective OPSD BoN-K requires Student token entropies from "
                "the pre-Teacher log-prob forward"
            )
        entropy_mean = (
            (entropies.detach() * response_mask).sum(dim=-1)
            / denom.to(entropies.device)
        ).cpu()
        eligible &= torch.isfinite(entropy_mean).to(eligible.device)

        selection = select_tlr_trajectories(
            prompt_ids=prompt_ids,
            eligible_mask=eligible,
            response_lengths=valid_counts,
            student_entropy_mean=entropy_mean,
            rollouts_per_prompt=config.rollouts_per_prompt,
            min_rollout_tokens=config.min_rollout_tokens,
        )
        selected_indices = selection.selected_indices
        if selected_indices.numel() == 0:
            # Every prompt group is degenerate (all rollouts below the floor):
            # no Teacher query for this batch at all. Historical bon4_lh
            # semantics: never fall back to ScoreLH over immediate-EOS.
            print(
                "[TLR] all prompt groups degenerate, skipping Teacher queries "
                f"this step (prompts={len(unique_prompt_ids)}, "
                f"min_rollout_tokens={config.min_rollout_tokens})"
            )
            return None, {
                "metrics": {
                    "tlr/enabled": 1.0,
                    "tlr/mode": TLR_MODE,
                    "tlr/num_prompts": float(len(unique_prompt_ids)),
                    "tlr/eligible_prompts": 0.0,
                    "tlr/no_eligible_prompts": float(len(unique_prompt_ids)),
                    "tlr/eligible_rollouts": 0.0,
                    "tlr/selected_rollouts": 0.0,
                    "tlr/target_teacher_queries": float(len(unique_prompt_ids)),
                    "tlr/actual_teacher_queries": 0.0,
                    "tlr/short_rollouts_lt16": float(int((valid_counts < 16).sum().item())),
                    "tlr/degenerate_rollouts": float(int((valid_counts < config.min_rollout_tokens).sum().item())),
                    "tlr/skipped_all_degenerate_prompts": float(len(unique_prompt_ids)),
                    "tlr/query_ratio_max_per_prompt": 1.0 / config.rollouts_per_prompt,
                    "tlr/query_ratio_sequence_actual": 0.0,
                    "tlr/query_ratio_among_eligible": 0.0,
                    "tlr/query_ratio_token_actual": 0.0,
                    "tlr/prompt_coverage": 0.0,
                },
                "records": [],
                "selection": selection,
            }
        short_rollout_count = int((valid_counts < 16).sum().item())
        degenerate_count = int((valid_counts < config.min_rollout_tokens).sum().item())
        skipped_prompts = len(unique_prompt_ids) - selection.selected_indices.numel()

        floor_eligible_cpu = selection.eligible_mask
        eligible_count = int(floor_eligible_cpu.sum().item())
        full_tokens = int(valid_counts[floor_eligible_cpu].sum().item())
        selected_tokens = int(valid_counts[selected_indices].sum().item())
        selected_batch = batch.select_idxs(selected_indices.numpy())

        selected_set = set(selected_indices.tolist())
        eligible_prompt_count = len(
            {prompt_ids[index] for index in torch.where(floor_eligible_cpu)[0].tolist()}
        )
        selected_per_prompt = torch.tensor(
            [
                sum(index in selected_set for index, uid in enumerate(prompt_ids) if uid == prompt_uid)
                for prompt_uid in unique_prompt_ids
            ],
            dtype=torch.float32,
        )
        selected_mask_cpu = torch.zeros(n_rollouts, dtype=torch.bool)
        selected_mask_cpu[selected_indices] = True
        unselected_eligible = floor_eligible_cpu & ~selected_mask_cpu
        chosen_lengths = valid_counts[selected_indices].float()
        unchosen_lengths = valid_counts[unselected_eligible].float()
        eligible_scores = selection.scores[floor_eligible_cpu]
        metrics = {
            "tlr/enabled": 1.0,
            "tlr/mode": TLR_MODE,
            "tlr/num_prompts": float(len(unique_prompt_ids)),
            "tlr/eligible_prompts": float(eligible_prompt_count),
            "tlr/no_eligible_prompts": float(len(unique_prompt_ids) - eligible_prompt_count),
            "tlr/eligible_rollouts": float(eligible_count),
            "tlr/selected_rollouts": float(selected_indices.numel()),
            "tlr/target_teacher_queries": float(len(unique_prompt_ids)),
            "tlr/actual_teacher_queries": float(selected_indices.numel()),
            "tlr/short_rollouts_lt16": float(short_rollout_count),
            "tlr/degenerate_rollouts": float(degenerate_count),
            "tlr/degenerate_rollout_ratio": degenerate_count / max(n_rollouts, 1),
            "tlr/skipped_all_degenerate_prompts": float(skipped_prompts),
            "tlr/query_ratio_max_per_prompt": 1.0 / config.rollouts_per_prompt,
            "tlr/query_ratio_sequence_actual": selected_indices.numel() / max(n_rollouts, 1),
            "tlr/query_ratio_among_eligible": selected_indices.numel() / max(eligible_count, 1),
            "tlr/query_ratio_token_actual": selected_tokens / max(full_tokens, 1),
            "tlr/prompt_coverage": selected_indices.numel() / max(len(unique_prompt_ids), 1),
            "tlr/selected_per_prompt_mean": float(selected_per_prompt.mean()),
            "tlr/selected_per_prompt_std": float(selected_per_prompt.std(unbiased=False)),
            "tlr/selected_per_prompt_min": float(selected_per_prompt.min()),
            "tlr/selected_per_prompt_max": float(selected_per_prompt.max()),
            "tlr/score_tlr_mean": float(eligible_scores.mean()),
            "tlr/score_tlr_std": float(eligible_scores.std(unbiased=False)),
            "tlr/score_tlr_min": float(eligible_scores.min()),
            "tlr/score_tlr_max": float(eligible_scores.max()),
            "tlr/full_eligible_response_tokens": float(full_tokens),
            "tlr/selected_response_tokens": float(selected_tokens),
            "tlr/selected_length_mean": float(chosen_lengths.mean()),
            "tlr/unselected_length_mean": (
                float(unchosen_lengths.mean()) if unchosen_lengths.numel() else float("nan")
            ),
            "tlr/selected_entropy_mean": float(entropy_mean[selected_indices].mean()),
            "tlr/unselected_entropy_mean": (
                float(entropy_mean[unselected_eligible].mean())
                if unselected_eligible.any()
                else float("nan")
            ),
        }
        source_indices = batch.non_tensor_batch.get("index", np.arange(n_rollouts))
        records = []
        rollout_seen: defaultdict[str, int] = defaultdict(int)
        for index, prompt_uid in enumerate(prompt_ids):
            rollout_idx = rollout_seen[prompt_uid]
            rollout_seen[prompt_uid] += 1
            is_selected = index in selected_set
            records.append(
                {
                    "run_name": str(self.config.trainer.experiment_name),
                    "seed": config.seed,
                    "global_step": self.global_steps,
                    "optimizer_step": self.global_steps,
                    "prompt_uid": prompt_uid,
                    "source_index": source_indices[index],
                    "rollout_idx": rollout_idx,
                    "eligible": int(eligible[index]),
                    "selected": int(is_selected),
                    "selection_mode": TLR_MODE,
                    "response_length": int(valid_counts[index]),
                    "response_valid_tokens": int(valid_counts[index]),
                    "student_entropy_mean": float(entropy_mean[index]),
                    "score_tlr": float(selection.scores[index]),
                    "teacher_queried": int(is_selected),
                    "actor_loss_contributed": int(is_selected),
                }
            )
        return selected_batch, {
            "metrics": metrics,
            "records": records,
            "selection": selection,
        }

    def _make_ff_prompt_uids(self, batch: DataProto) -> np.ndarray:
        # Before rollout the batch only contains input_ids/attention_mask/position_ids;
        # after rollout it also has a "prompts" key.  Accept either.
        if "prompts" in batch.batch.keys():
            prompt_ids = batch.batch["prompts"]
        else:
            prompt_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"][:, : prompt_ids.shape[-1]]
        source_indices = batch.non_tensor_batch.get("index", np.arange(len(batch)))
        data_sources = batch.non_tensor_batch.get("data_source", np.full(len(batch), "dataset", dtype=object))
        existing_ids = batch.non_tensor_batch.get("prompt_id", np.full(len(batch), None, dtype=object))
        result = []
        for row in range(len(batch)):
            valid_ids = prompt_ids[row][attention_mask[row].bool()].detach().cpu().tolist()
            prompt_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            result.append(
                make_prompt_uid(
                    str(data_sources[row]),
                    str(source_indices[row]),
                    prompt_text,
                    existing_id=existing_ids[row],
                )
            )
        return np.asarray(result, dtype=object)

    def _iter_ff_opd_batches(self):
        """Yield one fresh pass and, when configured, drain retry rounds."""
        # On resume, prompt_states contains the fresh prompts already consumed
        # and the StatefulDataLoader resumes at the next unseen fresh batch.
        fresh_seen: set[str] = set(self._ff_opd_manager.prompt_states)
        fresh_duplicates: set[str] = set()

        fresh_batch_count = 0
        for batch_dict in self.train_dataloader:
            if fresh_batch_count >= self.fresh_total_steps:
                break
            batch = DataProto.from_single_dict(batch_dict)
            batch.non_tensor_batch["uid"] = self._make_ff_prompt_uids(batch)
            batch.non_tensor_batch["ff_prompt_uid"] = batch.non_tensor_batch["uid"].copy()
            loaded_uids = [str(uid) for uid in batch.non_tensor_batch["ff_prompt_uid"]]
            fresh_duplicates.update(set(loaded_uids) & fresh_seen)
            fresh_seen.update(loaded_uids)
            for index, uid in enumerate(loaded_uids):
                stored = batch.select_idxs([index])
                # select_idxs shares meta_info by reference. Detach it so that
                # per-step keys written later in fit() (e.g. global_token_num at
                # line ~2117) cannot leak back into the store and break the
                # retry-time DataProto.concat consistency assertion when the
                # chunk mixes prompts from different fresh batches.
                stored.meta_info = dict(stored.meta_info)
                self._ff_prompt_store[uid] = stored
            fresh_batch_count += 1
            yield "fresh", batch, len(loaded_uids)

        dataset_prompt_count = len(self.train_dataset)
        missing = dataset_prompt_count - len(fresh_seen)
        _logger.info(
            "[FF coverage] dataset_unique_prompt_count=%d attempt1_unique_prompt_count=%d "
            "missing_fresh_prompt_count=%d duplicate_fresh_prompt_count=%d",
            dataset_prompt_count,
            len(fresh_seen),
            missing,
            len(fresh_duplicates),
        )
        if self._ff_opd_manager.config.debug_assertions and not self._ff_fresh_step_limited:
            assert missing == 0
            assert not fresh_duplicates

        if self._ff_opd_manager.config.max_no_success_retries == 0:
            return

        batch_size = int(self.config.data.get("gen_batch_size", self.config.data.train_batch_size))
        retry_round_active = bool(
            self._ff_opd_manager.retry_round > 0
            and self._ff_opd_manager.current_queue
        )
        if not retry_round_active:
            retry_round_active = self._ff_opd_manager.begin_retry_round()
        while retry_round_active:
            retry_round = self._ff_opd_manager.retry_round
            round_uids = list(self._ff_opd_manager.current_queue)
            if retry_round > self._ff_opd_manager.config.max_no_success_retries:
                raise AssertionError("FF-OPD exceeded its configured retry rounds")
            _logger.info(
                "[FF retry] round=%d prompt_count=%d",
                retry_round,
                len(round_uids),
            )
            for start in range(0, len(round_uids), batch_size):
                chunk_uids = round_uids[start : start + batch_size]
                missing_uids = [uid for uid in chunk_uids if uid not in self._ff_prompt_store]
                if missing_uids:
                    raise RuntimeError(f"FF-OPD prompt store is missing retry prompts: {missing_uids[:3]}")
                retry_batch = DataProto.concat([self._ff_prompt_store[uid] for uid in chunk_uids])
                yield f"retry_round_{retry_round}", retry_batch, len(chunk_uids)
            if self._ff_opd_manager.current_queue:
                raise AssertionError("each retry-round prompt must be consumed exactly once")
            retry_round_active = self._ff_opd_manager.begin_retry_round()

    @staticmethod
    def _drop_boundary_opd_payload(batch: DataProto) -> None:
        """Release selector-only hidden tensors before any Teacher dispatch."""

        for key in [
            name
            for name in list(batch.batch.keys())
            if name.startswith("boundary_")
        ]:
            batch.batch.pop(key)
        batch.non_tensor_batch.pop("ff_rollout_id", None)
        for key in [
            name
            for name in list(batch.meta_info)
            if name.startswith("boundary_")
        ]:
            batch.meta_info.pop(key, None)

    def _select_ff_opd_queries(self, batch: DataProto, epoch: int, ff_phase: str):
        """Verify siblings, route candidates, and crop before Teacher forward."""
        response_mask = batch.batch["response_mask"].bool()
        responses = batch.batch["responses"]
        eos_ids = self.tokenizer.eos_token_id
        eos_tensor = torch.as_tensor(eos_ids if isinstance(eos_ids, list) else [eos_ids], device=responses.device)
        has_eos = torch.isin(responses, eos_tensor).logical_and(response_mask).any(dim=-1)
        nonempty_response = response_mask.any(dim=-1)
        rollout_valid_tensor = has_eos & nonempty_response
        truncated_tensor = ~has_eos

        verifier_failed = False
        verifier_reward = torch.zeros_like(responses, dtype=torch.float32)
        verifier_extra = {}
        verifier_correct = [False] * len(batch)
        prevalid_indices = rollout_valid_tensor.nonzero(as_tuple=False).flatten().cpu().tolist()
        if prevalid_indices:
            try:
                verifier_batch = batch.select_idxs(prevalid_indices)
                valid_reward, verifier_extra = compute_reward(verifier_batch, self.reward_fn)
                verifier_reward[prevalid_indices] = valid_reward
                valid_scores = valid_reward.sum(dim=-1).detach().cpu()
                valid_correct = verifier_extra.get("acc")
                if valid_correct is None or len(valid_correct) != len(prevalid_indices):
                    valid_correct = [bool(score.item() > 0) for score in valid_scores]
                timeout_flags = verifier_extra.get("verifier_timeout", [False] * len(prevalid_indices))
                parse_flags = verifier_extra.get("verifier_parse_error", [False] * len(prevalid_indices))
                undetermined_flags = verifier_extra.get("verifier_undetermined", [False] * len(prevalid_indices))
                for local_index, batch_index in enumerate(prevalid_indices):
                    verifier_invalid = bool(timeout_flags[local_index] or parse_flags[local_index] or undetermined_flags[local_index])
                    if verifier_invalid:
                        rollout_valid_tensor[batch_index] = False
                    else:
                        verifier_correct[batch_index] = bool(valid_correct[local_index])
            except Exception as error:
                verifier_failed = True
                rollout_valid_tensor[prevalid_indices] = False
                verifier_extra = {"verifier_error": [str(error)] * len(prevalid_indices)}
        uids = batch.non_tensor_batch["ff_prompt_uid"]
        source_indices = batch.non_tensor_batch.get("index", np.arange(len(batch)))
        processing = batch.batch["attention_mask"].sum(dim=-1).detach().cpu().tolist()
        response_tokens = response_mask.sum(dim=-1).detach().cpu().tolist()
        if self._boundary_opd_enabled:
            if "ff_rollout_id" not in batch.non_tensor_batch:
                raise RuntimeError(
                    "Boundary-OPD requires stable ff_rollout_id before regroup"
                )
            grouped_items = stable_group_sibling_indices(
                uids,
                batch.non_tensor_batch["ff_rollout_id"],
                expected_siblings=self._ff_opd_manager.config.k_rollouts,
            )
        else:
            grouped: dict[str, list[int]] = defaultdict(list)
            for index, uid in enumerate(uids):
                grouped[str(uid)].append(index)
            grouped_items = list(grouped.items())
        prompts = []
        for uid, indices in grouped_items:
            if len(indices) != self._ff_opd_manager.config.k_rollouts:
                raise RuntimeError(
                    f"FF-OPD prompt {uid} has {len(indices)} siblings; expected "
                    f"{self._ff_opd_manager.config.k_rollouts}"
                )
            prompt = {
                "prompt_uid": uid,
                "source_index": str(source_indices[indices[0]]),
                "rollout_indices": indices,
                "verifier_correct": [int(bool(verifier_correct[index])) for index in indices],
                "rollout_valid": [bool(rollout_valid_tensor[index]) for index in indices],
                "processing_tokens": [processing[index] for index in indices],
                "response_tokens": [response_tokens[index] for index in indices],
                "response_masks": response_mask[indices],
                # Rollout already computed these sampled-token log-probs;
                # the selector only aligns their trajectories and adds no forward pass.
                "sampled_token_log_probs": batch.batch["old_log_probs"][indices],
                "student_token_entropies": (
                    batch.batch["entropys"][indices]
                    if "entropys" in batch.batch.keys()
                    else None
                ),
                "global_step": self.global_steps,
                "optimizer_step": self._ff_optimizer_step,
                "fresh_step": self._ff_fresh_step if ff_phase == "fresh" else "",
                "retry_step": self._ff_retry_step if ff_phase != "fresh" else "",
                "base_epoch": epoch + 1,
                "queue_source": ff_phase,
                "verifier_failed": verifier_failed,
                "truncated_rollout_count": sum(bool(truncated_tensor[index]) for index in indices),
                "verifier_timeout_count": sum(
                    bool(verifier_extra.get("verifier_timeout", [False] * len(indices))[local])
                    for local in range(min(len(indices), len(verifier_extra.get("verifier_timeout", []))))
                ),
                "verifier_parse_error_count": sum(
                    bool(verifier_extra.get("verifier_parse_error", [False] * len(indices))[local])
                    for local in range(min(len(indices), len(verifier_extra.get("verifier_parse_error", []))))
                ),
                "verifier_error_count": int(verifier_failed) * len(indices),
            }
            if self._boundary_opd_enabled:
                prompt.update(
                    {
                        "rollout_ids": [
                            int(batch.non_tensor_batch["ff_rollout_id"][index])
                            for index in indices
                        ],
                        "boundary_states": (
                            batch.batch["boundary_states"][indices]
                            if "boundary_states" in batch.batch.keys()
                            else None
                        ),
                        "boundary_transition_valid_mask": (
                            batch.batch["boundary_transition_valid_mask"][indices]
                            if "boundary_transition_valid_mask" in batch.batch.keys()
                            else None
                        ),
                        "boundary_hidden_capture_success": (
                            batch.batch["boundary_hidden_capture_success"][indices]
                            .detach()
                            .cpu()
                            .tolist()
                            if "boundary_hidden_capture_success" in batch.batch.keys()
                            else [False] * len(indices)
                        ),
                        "boundary_hidden_capture_error_code": (
                            batch.batch["boundary_hidden_capture_error_code"][indices]
                            .detach()
                            .cpu()
                            .tolist()
                            if "boundary_hidden_capture_error_code" in batch.batch.keys()
                            else [3] * len(indices)
                        ),
                        "boundary_hidden_capture_time_ms": (
                            batch.batch["boundary_hidden_capture_time_ms"][indices]
                            .detach()
                            .cpu()
                            .tolist()
                            if "boundary_hidden_capture_time_ms" in batch.batch.keys()
                            else [0.0] * len(indices)
                        ),
                        "boundary_communication_time_ms": (
                            batch.batch["boundary_communication_time_ms"][indices]
                            .detach()
                            .cpu()
                            .tolist()
                            if "boundary_communication_time_ms" in batch.batch.keys()
                            else [0.0] * len(indices)
                        ),
                        "boundary_extra_peak_memory_mb": (
                            batch.batch["boundary_extra_peak_memory_mb"][indices]
                            .detach()
                            .cpu()
                            .tolist()
                            if "boundary_extra_peak_memory_mb" in batch.batch.keys()
                            else [0.0] * len(indices)
                        ),
                        "boundary_hidden_capture_module": batch.meta_info.get(
                            "boundary_hidden_capture_module", ""
                        ),
                        "boundary_hidden_capture_method": batch.meta_info.get(
                            "boundary_hidden_capture_method", ""
                        ),
                        "boundary_hidden_dim": batch.meta_info.get(
                            "boundary_hidden_dim", 0
                        ),
                        "boundary_hidden_shape": batch.meta_info.get(
                            "boundary_hidden_shape", ()
                        ),
                        "boundary_sequence_parallel_size": batch.meta_info.get(
                            "boundary_sequence_parallel_size", 1
                        ),
                        "boundary_sequence_parallel_sharded": batch.meta_info.get(
                            "boundary_sequence_parallel_sharded", False
                        ),
                        "boundary_sequence_parallel_gathered": batch.meta_info.get(
                            "boundary_sequence_parallel_gathered", True
                        ),
                        "boundary_tensor_parallel_sharded": batch.meta_info.get(
                            "boundary_tensor_parallel_sharded", False
                        ),
                        "boundary_lm_head_module": batch.meta_info.get(
                            "boundary_lm_head_module", ""
                        ),
                        "boundary_lm_head_vocab_size": batch.meta_info.get(
                            "boundary_lm_head_vocab_size", 0
                        ),
                        "boundary_lm_head_weight_current": batch.meta_info.get(
                            "boundary_lm_head_weight_current", False
                        ),
                        "boundary_lm_head_vocab_sharded": batch.meta_info.get(
                            "boundary_lm_head_vocab_sharded", False
                        ),
                    }
                )
            prompts.append(prompt)
        result = self._ff_opd_manager.route_prompt_attempts(prompts)
        result.metrics.update(
            {
                "ff/fresh_step": float(self._ff_fresh_step),
                "ff/retry_step": float(self._ff_retry_step),
                "ff/optimizer_step": float(self._ff_optimizer_step),
            }
        )
        result.metrics["ff/teacher_verifier_errors"] = float(len(verifier_extra.get("verifier_error", [])))
        _logger.info(
            "[FF ablation] mode=%s frontier=%d target=%d actual=%d valid=%d "
            "correct=%d wrong=%d frontier_selected=%d nonfrontier_selected=%d ties=%d",
            self._ff_opd_manager.config.selector_mode,
            int(result.metrics["ff_ablation/frontier_prompt_count"]),
            int(result.metrics["ff_ablation/target_query_count"]),
            int(result.metrics["ff_ablation/actual_query_count"]),
            int(result.metrics["ff_ablation/selected_valid_count"]),
            int(result.metrics["ff_ablation/selected_correct_count"]),
            int(result.metrics["ff_ablation/selected_wrong_count"]),
            int(result.metrics["ff_ablation/selected_frontier_count"]),
            int(result.metrics["ff_ablation/selected_nonfrontier_count"]),
            int(result.metrics["ff_ablation/tie_count"]),
        )
        _logger.info(
            "[FF validity] valid_rollouts=%d correct_rollouts=%d incorrect_rollouts=%d "
            "invalid_rollouts=%d truncated_rollouts=%d verifier_timeouts=%d",
            sum(int(record["valid_rollout_count"]) for record in result.records),
            sum(int(record["valid_correct_count"]) for record in result.records),
            sum(int(record["valid_incorrect_count"]) for record in result.records),
            sum(int(record["invalid_count"]) for record in result.records),
            sum(int(record["truncated_count"]) for record in result.records),
            sum(int(record["verifier_timeout_count"]) for record in result.records),
        )
        _logger.info(
            "[FF selector] single_negative_direct_prompts=%d nearest_positive_selected_prompts=%d "
            "uninformative_fallback_prompts=%d all_pair_count=%d",
            int(result.metrics["ff/single_negative_direct_prompts"]),
            int(result.metrics["ff/nearest_positive_selected_prompts"]),
            int(result.metrics["ff/uninformative_fallback_prompts"]),
            int(result.metrics["ff/all_pair_count"]),
        )
        _logger.info(
            "[FF cost] comparable_prompts=%d switches=%d switch_rate=%.6f "
            "switch_rate_cumulative=%.6f selected_cost_mean=%.1f",
            int(result.metrics["ff/cost_comparable_prompts"]),
            int(result.metrics["ff/cost_switches"]),
            result.metrics["ff/cost_switch_rate"],
            result.metrics["ff/cost_switch_rate_cumulative"],
            result.metrics["ff/selected_candidate_cost_mean"],
        )
        _logger.info(
            "[FF buckets] all_correct_prompts=%d frontier_prompts=%d no_success_prompts=%d "
            "invalid_ambiguous_prompts=%d",
            int(result.metrics["ff/all_correct_prompts"]),
            int(result.metrics["ff/frontier_prompts"]),
            int(result.metrics["ff/no_success_prompts"]),
            int(result.metrics["ff/invalid_ambiguous_prompts"]),
        )
        _logger.info(
            "[FF query cap] available=%d used_this_step=%d used_total=%d "
            "full_opd_queries=%d cap=%d actual_ratio=%.6f",
            result.available_query_budget,
            result.used_queries,
            self._ff_opd_manager.used_queries,
            self._ff_opd_manager.total_full_opd_queries,
            int(result.metrics["ff/query_cap"]),
            result.metrics["ff/query_ratio_actual"],
        )
        _logger.info(
            "[FF analysis] distance_gap_mean=%.6f fallback_rate=%.6f fallback_rate_cumulative=%.6f "
            "no_success_to_frontier=%d no_success_to_frontier_rate_cumulative=%.6f query_ratio_actual=%.6f",
            result.metrics["ff/distance_gap_mean"],
            result.metrics["ff/fallback_rate"],
            result.metrics["ff/fallback_rate_cumulative"],
            int(result.metrics["ff/no_success_to_frontier"]),
            result.metrics["ff/no_success_to_frontier_rate_cumulative"],
            result.metrics["ff/query_ratio_actual"],
        )
        self._ff_opd_profiles.append(result.profile_records)
        if self._boundary_contrast_csv is not None:
            self._boundary_contrast_csv.append(result.boundary_csv_rows)
        if self._boundary_opd_enabled and self._ff_opd_manager.config.fresh_step_limit > 0:
            for row in result.boundary_csv_rows:
                _logger.info(
                    "[PDA candidate] step=%d prompt_id=%s rollout_id=%s area=%s "
                    "cost_ratio=%s final_score=%s representation_domain=%s "
                    "uses_hidden_difference=%s",
                    self.global_steps,
                    row["prompt_id"],
                    row["rollout_id"],
                    row["pda_area"],
                    row["cost_ratio"],
                    row["final_score"],
                    row["representation_domain"],
                    row["uses_hidden_difference"],
                )
        if not result.selected_indices:
            if self._boundary_opd_enabled:
                self._drop_boundary_opd_payload(batch)
            self._ff_opd_csv.append(result.records)
            return None, result, verifier_reward
        selected = batch.select_idxs(result.selected_indices)
        if self._boundary_opd_enabled:
            # Selection has completed; neither the Teacher nor Actor update may
            # receive Boundary routing tensors.
            self._drop_boundary_opd_payload(selected)
            self._drop_boundary_opd_payload(batch)
        selected.non_tensor_batch["ff_selected_sibling_index"] = np.asarray(
            [candidate.frontier_selection.selected_negative_idx for candidate in result.candidates],
            dtype=np.int64,
        )
        selected.meta_info["ff_opd_selected_candidate_count"] = len(result.selected_indices)
        selected.meta_info["ff_opd_available_query_budget"] = result.available_query_budget
        if result.ht_enabled:
            if len(result.candidates) != len(selected):
                raise AssertionError("L8-HT candidates must align one-to-one with the Teacher batch")
            device = selected.batch["response_mask"].device
            selected.batch["ff_opd_query_probability"] = torch.tensor(
                [candidate.query_probability for candidate in result.candidates],
                dtype=torch.float64,
                device=device,
            )
            selected.batch["ff_opd_query_weight"] = torch.tensor(
                [candidate.inverse_probability_weight for candidate in result.candidates],
                dtype=torch.float64,
                device=device,
            )
            selected.meta_info.update(
                {
                    "ff_opd_ht_enabled": True,
                    "ff_opd_ht_full_valid_response_tokens": result.ht_full_valid_response_tokens,
                    "ff_opd_ht_full_candidate_count": result.ht_full_candidate_count,
                    "ff_opd_ht_real_queried_rollout_count": len(selected),
                }
            )
        if len(selected) != len(result.selected_indices):
            raise AssertionError("FF-OPD Teacher batch size differs from selected candidate count")
        if self._ff_opd_manager.config.debug_assertions:
            assert all(
                record["selected_rollout_valid"] == 1
                for record in result.records
                if record["selected_for_teacher"]
            )
            if self._ff_opd_manager.config.selector_mode != "global_random":
                assert all(
                    record["current_bucket"] == "frontier"
                    for record in result.records
                    if record["selected_for_teacher"]
                )
            if self._ff_opd_manager.config.selector_mode not in {"global_random", "random_correct"}:
                assert all(
                    record["selected_rollout_correct"] == 0
                    for record in result.records
                    if record["selected_for_teacher"]
                )
            elif self._ff_opd_manager.config.selector_mode == "random_correct":
                assert all(
                    record["selected_rollout_correct"] == 1
                    for record in result.records
                    if record["selected_for_teacher"]
                )
        return selected, result, verifier_reward

    def _finalize_ff_opd_records(
        self, batch: DataProto, route_result, teacher_forward_seconds: float = 0.0
    ) -> None:
        selected_records = {
            (record["prompt_uid"], int(record["selected_rollout_idx"])): record
            for record in route_result.records
            if record["selected_for_teacher"]
        }
        sampled_reward = batch.batch.get("sampled_opd_rm_scores", batch.batch.get("rm_scores"))
        if sampled_reward is None:
            raise RuntimeError("FF-OPD Teacher output did not contain the existing sampled-token OPD target")
        if sampled_reward.requires_grad:
            raise AssertionError("FF-OPD Teacher sampled-token target must be stop-gradient")
        student_logp = batch.batch["old_log_probs"]
        teacher_logp = (student_logp + sampled_reward).detach()
        if teacher_logp.requires_grad:
            raise AssertionError("FF-OPD Teacher sampled log-prob must be stop-gradient")
        selected_siblings = batch.non_tensor_batch["ff_selected_sibling_index"]
        kl_profile_rows = []
        for row, uid in enumerate(batch.non_tensor_batch["ff_prompt_uid"]):
            record = selected_records[(str(uid), int(selected_siblings[row]))]
            mask = batch.batch["response_mask"][row]
            stats = sampled_reverse_kl_statistics(student_logp[row], teacher_logp[row], mask)
            supervised = int(mask.sum().item())
            record.update(stats)
            if self._ff_kl_profiles is not None:
                # Position-resolved training signal: r_t on supervised tokens.
                ratio = (student_logp[row] - teacher_logp[row])[mask]
                kl_profile_rows.append(
                    {
                        "run_name": str(self.config.trainer.experiment_name),
                        "prompt_uid": record["prompt_uid"],
                        "attempt_id": record.get("attempt_id", 1),
                        "global_step": self.global_steps,
                        "selected_rollout_idx": record.get("selected_rollout_idx"),
                        "response_token_count": supervised,
                        "reverse_kl_tokens": [
                            round(float(value), 6)
                            for value in ratio.detach().float().cpu().tolist()
                        ],
                    }
                )
            state = self._ff_opd_manager.prompt_states[record["prompt_uid"]]
            state.total_teacher_queries += 1
            record.update(
                teacher_processing_tokens=int(batch.batch["attention_mask"][row].sum().item()),
                teacher_scored_tokens=supervised,
                teacher_supervised_tokens=supervised,
                response_mask_token_count=supervised,
                opd_loss_contributed=int(supervised > 0),
                teacher_query_count_total=state.total_teacher_queries,
                used_queries=self._ff_opd_manager.used_queries,
                # Wall-clock of the synchronized Teacher forward for this step;
                # every selected row of the step shares the same measurement.
                teacher_forward_seconds=float(teacher_forward_seconds),
                reverse_kl_mean=stats["sampled_reverse_kl_token_mean"],
            )
            state.cumulative_teacher_processing_tokens += record["teacher_processing_tokens"]
            state.cumulative_teacher_supervised_tokens += supervised
            state.last_optimizer_step = self._ff_optimizer_step + 1
            state.last_student_model_version = self._ff_opd_manager.student_model_version
        if self._ff_kl_profiles is not None and kl_profile_rows:
            self._ff_kl_profiles.append(kl_profile_rows)

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        gradient_diagnostics_config = self.config.trainer.rollout_gradient_diagnostics
        gradient_diagnostics_enabled = bool(gradient_diagnostics_config.enable)
        if gradient_diagnostics_enabled:
            if self._ff_opd_enabled or self._tlr_opd_enabled or self.config.actor_rollout_ref.rollout.get(
                "ta_opd_enable", False
            ):
                raise RuntimeError("rollout gradient diagnostics must run on the unmodified Full OPD path")
            if self.config.algorithm.adv_estimator != "token_reward_direct":
                raise RuntimeError("rollout gradient diagnostics require algorithm.adv_estimator=token_reward_direct")
            if self.config.actor_rollout_ref.actor.ppo_epochs != 1:
                raise RuntimeError("rollout gradient diagnostics require actor.ppo_epochs=1")
            if self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0) != 0:
                raise RuntimeError("rollout gradient diagnostics require sampled-token Full OPD (log_prob_top_k=0)")
            if self.config.actor_rollout_ref.actor.strategy != "fsdp":
                raise RuntimeError("rollout gradient diagnostics currently require the Full OPD FSDP actor")
            if not bool(self.config.reward_model.get("compute_teacher_entropy", True)):
                raise RuntimeError("rollout gradient diagnostics require reward_model.compute_teacher_entropy=True")
            if gradient_diagnostics_config.save_per_rollout and not gradient_diagnostics_config.save_dir:
                raise RuntimeError(
                    "trainer.rollout_gradient_diagnostics.save_dir is required when save_per_rollout=true"
                )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # FF-OPD has a fixed fresh horizon and queue-driven retry work. Keep
        # their progress displays separate instead of advertising a fake
        # fresh+maximum-retries total.
        if self._ff_opd_enabled:
            progress_bar = tqdm(
                total=self.fresh_total_steps,
                initial=self._ff_fresh_step,
                desc=f"Fresh {self.fresh_total_steps} steps",
            )
            ff_progress_phase = "fresh"
        else:
            progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
            ff_progress_phase = None

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        ff_dataset_prompt_count = len(self.train_dataset) if self._ff_opd_enabled else 0
        if self._ff_opd_enabled:
            _logger.info("[FF startup] dataset_unique_prompt_count=%d drop_last=false", ff_dataset_prompt_count)
        # Non-FF-OPD dataloader is a fixed-size, reused object; its length is
        # the exact optimizer-step count of one dataset epoch, known only
        # after prompt tokenization/length filtering, so it can't be guessed
        # from the raw dataset row count ahead of time.
        general_steps_per_epoch = 0 if self._ff_opd_enabled else len(self.train_dataloader)
        self._epoch_step_idx = 0
        for epoch in range(self.config.trainer.total_epochs):
            if self._ff_opd_enabled and epoch > 0:
                # New dataset epoch: re-run the fresh pass with the current
                # student. Clear the per-epoch prompt classification, retry
                # queues, and prompt store so the fresh-coverage/disjointness
                # assertions hold again, and restart the fresh counter and
                # progress bar so this epoch's end-of-pass checkpoint fires at
                # its final fresh step (global_step_{k*fresh_total_steps}).
                # Cumulative counters and student_model_version are preserved.
                self._ff_opd_manager.begin_new_epoch()
                self._ff_prompt_store = {}
                self._ff_fresh_step = 0
                progress_bar.close()
                progress_bar = tqdm(
                    total=self.fresh_total_steps,
                    initial=0,
                    desc=f"Fresh {self.fresh_total_steps} steps (epoch {epoch + 1})",
                )
                ff_progress_phase = "fresh"
            if self._ff_opd_enabled:
                training_batches = self._iter_ff_opd_batches()
            else:
                self._epoch_step_idx = 0
                training_batches = (
                    ("standard", DataProto.from_single_dict(batch_dict), len(batch_dict))
                    for batch_dict in self.train_dataloader
                )
            for ff_phase, batch, loaded_prompt_count in training_batches:
                if self._ff_opd_enabled and ff_phase != ff_progress_phase:
                    progress_bar.close()
                    retry_round = self._ff_opd_manager.retry_round
                    retry_batch_size = int(
                        self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
                    )
                    retry_batches = (
                        len(self._ff_opd_manager.current_queue) + retry_batch_size - 1
                    ) // retry_batch_size
                    progress_bar = tqdm(
                        total=retry_batches,
                        desc=(
                            f"Retry round {retry_round}/"
                            f"{self._ff_opd_manager.config.max_no_success_retries}"
                        ),
                    )
                    ff_progress_phase = ff_phase

                def advance_batch_counters_and_progress() -> None:
                    if self._ff_opd_enabled:
                        if ff_phase == "fresh":
                            self._ff_fresh_step += 1
                        else:
                            self._ff_retry_step += 1
                    else:
                        self._epoch_step_idx += 1
                    progress_bar.update(1)

                metrics = {
                    "ff/enabled": float(self._ff_opd_enabled),
                    "tlr/enabled": float(self._tlr_opd_enabled),
                }
                timing_raw = {}
                ff_opd_context = None
                tlr_opd_context = None

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                # add uid to batch
                if self._ff_opd_enabled:
                    generated_prompt_uids = {str(uid) for uid in batch.non_tensor_batch["ff_prompt_uid"]}
                    if self._ff_opd_manager.config.debug_assertions and ff_phase == "fresh":
                        assert generated_prompt_uids.isdisjoint(self._ff_opd_manager.prompt_states)
                    _logger.info(
                        "[FF pre-rollout] phase=%s loaded_prompts=%d student_prompt_count=%d "
                        "student_rollout_count=%d model_version=%s",
                        ff_phase,
                        loaded_prompt_count,
                        len(generated_prompt_uids),
                        len(generated_prompt_uids) * self._ff_opd_manager.config.k_rollouts,
                        self._ff_opd_manager.student_model_version,
                    )
                else:
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                if gradient_diagnostics_enabled:
                    batch.meta_info["rollout_gradient_diagnostics_enable"] = True
                    batch.meta_info["rollout_gradient_diagnostics_eps"] = float(gradient_diagnostics_config.eps)

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                # Padding for FF-OPD: ensure batch size is divisible by DP size
                # This is necessary because FF-OPD uses drop_last=False, which can result in
                # incomplete batches that cannot be evenly divided among DP ranks.
                # The padding is removed again right after generation so the repeated
                # prompt-level batch still aligns with the rollout-level output.
                if self._ff_opd_enabled:
                    dp_size = self.actor_rollout_wg.world_size
                    _gen_pre_pad_size = len(gen_batch_output)
                    gen_batch_output, _gen_pad_size = pad_dataproto_to_divisor(gen_batch_output, dp_size)
                    if _gen_pad_size:
                        _logger.info(
                            "[FF padding] batch_size=%d dp_size=%d padding_size=%d",
                            _gen_pre_pad_size,
                            dp_size,
                            _gen_pad_size,
                        )

                # Queue exhaustion, rather than the fresh scheduler horizon,
                # terminates FF-OPD retry work.
                is_last_step = (
                    not self._ff_opd_enabled and self.global_steps >= self.total_training_steps
                )
                step_wall_start = time.perf_counter()
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                        # Remove FF-OPD generation padding before aligning with the
                        # prompt-level batch.
                        if self._ff_opd_enabled:
                            gen_batch_output = unpad_dataproto(gen_batch_output, _gen_pad_size)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            
                            # Padding for FF-OPD: ensure batch size is divisible by DP size
                            if self._ff_opd_enabled:
                                dp_size = self.actor_rollout_wg.world_size
                                batch_size = len(gen_baseline_batch)
                                if batch_size % dp_size != 0:
                                    padding_size = dp_size - (batch_size % dp_size)
                                    gen_baseline_batch.padding(padding_size=padding_size, padding_candidate="last")
                            
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                # pass global_steps and is_plot config to rm_wg
                                batch.meta_info["global_steps"] = self.global_steps
                                batch.meta_info["is_plot"] = self.config.trainer.get("is_plot", False)
                                _baseline_dp = int(getattr(self.rm_wg, "world_size", 1))
                                batch = filter_truncated_rollouts(
                                    batch,
                                    eos_token_id=self.tokenizer.eos_token_id,
                                    dp_size=_baseline_dp,
                                )
                                if len(batch) > 0:
                                    # Padding for FF-OPD: ensure batch size is divisible by DP size
                                    if self._ff_opd_enabled:
                                        dp_size = self.rm_wg.world_size
                                        batch_size = len(batch)
                                        if batch_size % dp_size != 0:
                                            padding_size = dp_size - (batch_size % dp_size)
                                            batch.padding(padding_size=padding_size, padding_candidate="last")
                                    
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                    batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    if self._boundary_opd_enabled:
                        num_siblings = self._ff_opd_manager.config.k_rollouts
                        if len(batch) % num_siblings:
                            raise RuntimeError(
                                "Boundary-OPD rollout batch is not divisible by K"
                            )
                        # Assigned before DP padding and token-count balancing so
                        # the identity follows each row through every reorder.
                        batch.non_tensor_batch["ff_rollout_id"] = np.tile(
                            np.arange(num_siblings, dtype=np.int64),
                            len(batch) // num_siblings,
                        )
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # FF-OPD tail batches are produced with drop_last=False, so the rollout
                    # batch size may not be divisible by the Actor DP world size (e.g. 244
                    # rollouts with DP=8). Pad to the next DP multiple and mark the synthetic
                    # rows so that seqlen balancing and the student first-forward both see a
                    # DP-divisible batch. The pad rows are stripped again right before FF
                    # query selection so the selector only ever sees real rollouts.
                    _ff_dp_pad_count = 0
                    if self._ff_opd_enabled:
                        _actor_world = int(self.actor_rollout_wg.world_size)
                        _actor_sp = int(self.config.actor_rollout_ref.actor.ulysses_sequence_parallel_size)
                        _actor_dp = _actor_world // _actor_sp
                        if _actor_world % _actor_sp:
                            raise RuntimeError("Actor world size must be divisible by Ulysses SP size")
                        _ff_dp_pad_count = (-len(batch)) % _actor_dp
                        if _ff_dp_pad_count:
                            _pre_pad = len(batch)
                            batch, _ff_dp_pad_count = pad_dataproto_to_divisor(batch, _actor_dp)
                            _pad_mask = torch.zeros(
                                len(batch), dtype=torch.bool, device=batch.batch["attention_mask"].device
                            )
                            _pad_mask[_pre_pad:] = True
                            batch.batch["_ff_dp_pad"] = _pad_mask
                            _logger.info(
                                "[FF DP alignment] balance/first-forward pad: batch_before=%d pad=%d "
                                "batch_after=%d actor_dp=%d",
                                _pre_pad,
                                _ff_dp_pad_count,
                                len(batch),
                                _actor_dp,
                            )

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    teacher_call_latency_sec = None
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            with marked_timer("compute_log_prob", timing_raw, color="blue"):
                                # First forward, get student top k ids and log probs
                                print("First forward, get student top k ids and log probs")
                                # Ordinary TLR uses only mean Student token entropy;
                                # no verifier labels or hidden-state router features.
                                batch.meta_info.pop("tlr_opd_enable", None)
                                if self._boundary_hidden_capture_enabled:
                                    # Keep a selector-only mask whose valid
                                    # tokens are response content: no prompt,
                                    # assistant template, PAD, or EOS.
                                    eos_ids = self.tokenizer.eos_token_id
                                    eos_values = eos_ids if isinstance(eos_ids, list) else [eos_ids]
                                    eos_values = [value for value in eos_values if value is not None]
                                    boundary_response_mask = batch.batch["response_mask"].bool().clone()
                                    if eos_values:
                                        eos_tensor = torch.as_tensor(
                                            eos_values,
                                            device=batch.batch["responses"].device,
                                        )
                                        boundary_response_mask &= ~torch.isin(
                                            batch.batch["responses"], eos_tensor
                                        )
                                    batch.batch["boundary_response_mask"] = boundary_response_mask
                                    batch.meta_info["boundary_opd_enable"] = True
                                    batch.meta_info[
                                        "boundary_opd_config"
                                    ] = self._boundary_opd_worker_config
                                if self.config.actor_rollout_ref.rollout.get("ta_opd_enable", False):
                                    batch.meta_info["selector_top_k"] = self.config.actor_rollout_ref.rollout.get(
                                        "ta_opd_topk", 16
                                    )
                                    batch.meta_info["selector_top_k_strategy"] = (
                                        self.config.actor_rollout_ref.rollout.get(
                                            "ta_opd_score_top_k_strategy", "union"
                                        )
                                    )
                                # Pop the FF pad marker before dispatch so the synthetic column
                                # is not sent to the workers. The marker stays aligned with the
                                # (balanced) batch order, which compute_log_prob preserves, so it
                                # can be used to drop the pad rows right after.
                                _ff_pad_mask = None
                                if "_ff_dp_pad" in batch.batch.keys():
                                    _ff_pad_mask = batch.batch.pop("_ff_dp_pad")
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

                                # if "entropys" in old_log_prob.batch.keys():
                                #    old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)

                            if self._boundary_calibration is not None:
                                calibration_batch = batch
                                if _ff_pad_mask is not None and bool(_ff_pad_mask.any()):
                                    calibration_batch = batch.select_idxs((~_ff_pad_mask).nonzero(as_tuple=True)[0])
                                self._boundary_calibration.update(
                                    calibration_batch.batch["boundary_states"],
                                    calibration_batch.batch["boundary_transition_valid_mask"],
                                    calibration_batch.non_tensor_batch["ff_prompt_uid"],
                                    calibration_batch.non_tensor_batch["ff_rollout_id"],
                                )
                                if self._boundary_calibration.complete:
                                    artifact = self._boundary_calibration.finalize()
                                    _logger.info(
                                        "Boundary calibration complete: manifest=%s sha256=%s samples=%d hidden_dim=%d",
                                        self._ff_opd_manager.config.boundary_opd.calibration_artifact_path,
                                        artifact.manifest_sha256,
                                        artifact.manifest["sample_count"],
                                        artifact.manifest["hidden_dim"],
                                    )
                                    progress_bar.close()
                                    self._ff_opd_csv.close()
                                    self._ff_opd_profiles.close()
                                    if self._ff_kl_profiles is not None:
                                        self._ff_kl_profiles.close()
                                    if self._boundary_contrast_csv is not None:
                                        self._boundary_contrast_csv.close()
                                    return
                                advance_batch_counters_and_progress()
                                self.global_steps += 1
                                continue

                            if self._tlr_opd_enabled:
                                batch, tlr_opd_context = self._select_tlr_queries(batch)
                                metrics.update(tlr_opd_context["metrics"])
                                if batch is None:
                                    _logger.warning(
                                        "[TLR] no prompt has an eligible rollout; skipping Teacher and Actor"
                                    )
                                    metrics["training/global_step"] = self.global_steps
                                    metrics["training/epoch"] = epoch
                                    logger.log(data=metrics, step=self.global_steps)
                                    advance_batch_counters_and_progress()
                                    self.global_steps += 1
                                    continue
                            if self._ff_opd_enabled:
                                # Drop the synthetic DP-alignment pad rows before FF query
                                # selection so the selector only ever sees real rollouts.
                                if _ff_pad_mask is not None and bool(_ff_pad_mask.any()):
                                    _keep = (~_ff_pad_mask).nonzero(as_tuple=True)[0]
                                    batch = batch.select_idxs(_keep)
                                batch, ff_opd_context, _ = self._select_ff_opd_queries(batch, epoch, ff_phase)
                                metrics.update(ff_opd_context.metrics)
                                if batch is None:
                                    metrics["ff/actual_optimizer_steps"] = 0.0
                                    metrics["training/global_step"] = self.global_steps
                                    metrics["training/epoch"] = epoch
                                    empty_step_time = time.perf_counter() - step_wall_start
                                    metrics.update(
                                        {
                                            "cost/teacher_processed_input_token_count": 0.0,
                                            "cost/teacher_scored_response_token_count": 0.0,
                                            "cost/teacher_latency_sec": 0.0,
                                            "perf/time_per_step": empty_step_time,
                                            "timing_s/step": empty_step_time,
                                        }
                                    )
                                    self._write_opd_step_metrics(
                                        metrics, self.resource_pool_manager.get_n_gpus()
                                    )
                                    logger.log(data=metrics, step=self.global_steps)
                                    advance_batch_counters_and_progress()
                                    self.global_steps += 1
                                    if is_last_step:
                                        progress_bar.close()
                                        self._ff_opd_csv.close()
                                        self._ff_opd_profiles.close()
                                        if self._ff_kl_profiles is not None:
                                            self._ff_kl_profiles.close()
                                        return
                                    continue

                            # Get Top-K parameters from config
                            loss_top_k = self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0)
                            loss_strategy = self.config.actor_rollout_ref.rollout.get(
                                "top_k_strategy", "only_stu"
                            )
                            ta_selector_enabled = self.config.actor_rollout_ref.rollout.get(
                                "ta_opd_enable", False
                            )
                            if ta_selector_enabled:
                                top_k = self.config.actor_rollout_ref.rollout.get("ta_opd_topk", 16)
                                strategy = self.config.actor_rollout_ref.rollout.get(
                                    "ta_opd_score_top_k_strategy", "union"
                                )
                            else:
                                top_k = loss_top_k
                                strategy = loss_strategy
                            kl_estimator = self.config.actor_rollout_ref.rollout.get("kl_estimator", "k1")
                            reward_weight_mode = self.config.actor_rollout_ref.rollout.get(
                                "reward_weight_mode", "student_p"
                            )

                            # pass global_steps and is_plot config to rm_wg
                            batch.meta_info["global_steps"] = self.global_steps
                            batch.meta_info["is_plot"] = self.config.trainer.get("is_plot", False)
                            teacher_temperature = self.config.actor_rollout_ref.rollout.get("teacher_temperature", 1.0)

                            batch.meta_info["log_prob_top_k"] = loss_top_k
                            batch.meta_info["top_k_strategy"] = loss_strategy
                            if ta_selector_enabled:
                                batch.meta_info["selector_top_k"] = top_k
                                batch.meta_info["selector_top_k_strategy"] = strategy
                            else:
                                batch.meta_info.pop("selector_top_k", None)
                                batch.meta_info.pop("selector_top_k_strategy", None)
                            batch.meta_info["kl_estimator"] = kl_estimator
                            batch.meta_info["reward_weight_mode"] = reward_weight_mode
                            batch.meta_info["teacher_temperature"] = teacher_temperature
                            batch.meta_info["ta_opd_enable"] = self.config.actor_rollout_ref.rollout.get(
                                "ta_opd_enable", False
                            )
                            batch.meta_info["tlr_privileged_teacher"] = self._tlr_opd_enabled
                            # Inject micro-batch multiplier so Teacher workers can process more
                            # tokens per GPU forward pass.  Configured via:
                            #   reward_model:
                            #     micro_batch_multiplier: 2   # default: 1 (no change)
                            batch.meta_info["rm_micro_batch_multiplier"] = int(
                                self.config.reward_model.get("micro_batch_multiplier", 1)
                            )
                            # ------------------------------------------------------------------ #
                            # Teacher routing: filter truly invalid rollouts, then pad with      #
                            # dummy samples to satisfy Teacher DP divisibility.                  #
                            #                                                                    #
                            # Principle: valid candidates must NEVER be discarded to satisfy a   #
                            # DP alignment constraint.  Instead we add dummy samples that are    #
                            # masked out of all loss / statistics computations.                  #
                            # ------------------------------------------------------------------ #
                            _teacher_dp = int(getattr(self.rm_wg, "world_size", 1))

                            # Step 1: filter genuinely invalid rollouts (truncated / overlength).
                            batch, _teacher_filter_stats = filter_truncated_rollouts(
                                batch,
                                eos_token_id=self.tokenizer.eos_token_id,
                                dp_size=_teacher_dp,
                                allow_truncated=self._tlr_opd_enabled,
                            )
                            _n_real = _teacher_filter_stats.n_real
                            if self._tlr_opd_enabled and tlr_opd_context is not None:
                                expected_tlr_queries = int(
                                    tlr_opd_context["metrics"]["tlr/actual_teacher_queries"]
                                )
                                if _n_real != expected_tlr_queries:
                                    raise RuntimeError(
                                        "TLR Teacher query mismatch after routing: "
                                        f"selected={expected_tlr_queries} real_teacher_rows={_n_real} "
                                        f"dropped={_teacher_filter_stats.n_before - _n_real} "
                                        "(selected rollouts were dropped by the Teacher filter)"
                                    )
                            if batch.meta_info.get("ff_opd_ht_enabled", False) and (
                                _n_real != _teacher_filter_stats.n_before
                            ):
                                raise RuntimeError(
                                    "L8-HT cannot drop a sampled rollout after categorical selection; "
                                    "doing so would invalidate its inclusion probability"
                                )
                            if self._ff_opd_enabled and self._ff_opd_manager.config.debug_assertions:
                                assert _teacher_filter_stats.drop_truncated == 0, (
                                    "FF-OPD selected a truncated rollout after pre-selector validity filtering"
                                )

                            # Emit structured routing log.
                            _logger.info(
                                "[Teacher routing] candidates=%d  overlength_dropped=%d  "
                                "truncated_dropped=%d  invalid_dropped=%d  real_selected=%d",
                                _teacher_filter_stats.n_before,
                                _teacher_filter_stats.drop_overlength,
                                _teacher_filter_stats.drop_truncated,
                                _teacher_filter_stats.drop_invalid,
                                _n_real,
                            )
                            # Emit FF-OPD bucket distribution (all_correct / frontier / no_success).
                            if self._ff_opd_enabled and ff_opd_context is not None:
                                _m = ff_opd_context.metrics
                                _logger.info(
                                    "[FF-OPD buckets] all_correct=%d  frontier=%d  no_success=%d  "
                                    "frontier_selected=%d  no_success_selected=%d",
                                    int(_m.get("ff/all_correct_prompts", 0)),
                                    int(_m.get("ff/frontier_prompts", 0)),
                                    int(_m.get("ff/no_success_prompts", 0)),
                                    int(_m.get("ff/frontier_selected", 0)),
                                    int(_m.get("ff/no_success_selected", 0)),
                                )
                            if _teacher_filter_stats.drop_truncated > 0:
                                _logger.warning(
                                    "[Teacher routing] %d/%d rollouts were truncated (no EOS); "
                                    "they are excluded from Teacher forward.",
                                    _teacher_filter_stats.drop_truncated,
                                    _teacher_filter_stats.n_before,
                                )

                            # Keep FF-OPD bookkeeping consistent after filtering.
                            if self._ff_opd_enabled and "ff_opd_selected_candidate_count" in batch.meta_info:
                                batch.meta_info["ff_opd_selected_candidate_count"] = _n_real

                            if _n_real == 0:
                                _logger.warning(
                                    "[Teacher routing] All rollouts were invalid; "
                                    "skipping Teacher call for this step."
                                )
                                metrics["training/global_step"] = self.global_steps
                                metrics["training/epoch"] = epoch
                                empty_step_time = time.perf_counter() - step_wall_start
                                metrics.update(
                                    {
                                        "cost/teacher_processed_input_token_count": 0.0,
                                        "cost/teacher_scored_response_token_count": 0.0,
                                        "cost/teacher_latency_sec": 0.0,
                                        "perf/time_per_step": empty_step_time,
                                        "timing_s/step": empty_step_time,
                                    }
                                )
                                self._write_opd_step_metrics(
                                    metrics, self.resource_pool_manager.get_n_gpus()
                                )
                                logger.log(data=metrics, step=self.global_steps)
                                advance_batch_counters_and_progress()
                                self.global_steps += 1
                                if is_last_step:
                                    progress_bar.close()
                                    return
                                continue

                            # Step 2: pad with dummy samples to satisfy Teacher DP alignment.
                            batch, _teacher_filter_stats = pad_teacher_batch_with_dummy(
                                batch, dp_size=_teacher_dp, stats=_teacher_filter_stats
                            )
                            _n_dummy = _teacher_filter_stats.n_dummy
                            _n_padded = _teacher_filter_stats.n_padded

                            if _n_dummy > 0:
                                _logger.info(
                                    "[Teacher routing] dummy_padded=%d  teacher_batch=%d  teacher_dp=%d",
                                    _n_dummy,
                                    _n_padded,
                                    _teacher_dp,
                                )

                            # Assertions: DP alignment must hold; no valid candidate was dropped.
                            assert _n_padded % _teacher_dp == 0, (
                                f"Teacher batch size {_n_padded} is not divisible by DP={_teacher_dp}"
                            )
                            if self._ff_opd_enabled and self._ff_opd_manager.config.debug_assertions:
                                assert _n_real + _n_dummy == _n_padded
                                if _n_dummy:
                                    assert batch.batch["response_mask"][_n_real:].sum().item() == 0
                                assert _teacher_filter_stats.drop_truncated == 0
                            assert _teacher_filter_stats.drop_overlength + _teacher_filter_stats.drop_truncated + \
                                _teacher_filter_stats.drop_invalid == \
                                _teacher_filter_stats.n_before - _n_real, (
                                "Teacher filter drop counts are inconsistent"
                            )

                            with marked_timer("compute_rm_score", timing_raw, color="magenta"):
                                teacher_data = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(teacher_data)

                            # Step 3: strip dummy samples from Teacher output before any
                            # downstream computation (loss, statistics, selector updates).
                            if _n_dummy > 0:
                                batch = strip_dummy_samples(batch, n_real=_n_real)
                                _logger.debug(
                                    "[Teacher routing] Stripped %d dummy samples; "
                                    "real batch size restored to %d.",
                                    _n_dummy,
                                    _n_real,
                                )

                            if self._ff_opd_enabled:
                                expected = int(batch.meta_info["ff_opd_selected_candidate_count"])
                                if len(batch) != expected:
                                    raise AssertionError(
                                        f"FF-OPD selected {expected} candidates but Teacher returned {len(batch)} "
                                        f"(after stripping {_n_dummy} dummy samples)"
                                    )

                            # The Ray call above returns only after the Teacher
                            # result is materialized, so this covers the full
                            # synchronized Teacher forward rather than enqueue time.
                            teacher_call_latency_sec = timing_raw["compute_rm_score"]

                            if self._tlr_opd_enabled and tlr_opd_context is not None:
                                response_mask_tlr = batch.batch["response_mask"].bool()
                                sampled_reward = batch.batch.get("sampled_opd_rm_scores")
                                if sampled_reward is None:
                                    sampled_reward = batch.batch.get("rm_scores")
                                if sampled_reward is None or sampled_reward.dim() != 2:
                                    raise RuntimeError(
                                        "TLR requires the existing sampled-token [B,T] Teacher reward"
                                    )
                                sampled_logratio = -sampled_reward.detach()
                                selection = tlr_opd_context["selection"]
                                selected_indices = selection.selected_indices
                                selected_reverse_kl_mean = float(
                                    (sampled_logratio * response_mask_tlr).sum()
                                    / response_mask_tlr.sum().clamp_min(1)
                                )
                                input_tokens_by_row = (
                                    batch.batch["attention_mask"].sum(dim=-1).detach().cpu()
                                )
                                scored_tokens_by_row = response_mask_tlr.sum(dim=-1).detach().cpu()
                                input_tokens = int(input_tokens_by_row.sum())
                                scored_tokens = int(scored_tokens_by_row.sum())
                                teacher_hours = teacher_call_latency_sec * _teacher_dp / 3600.0
                                metrics.update(
                                    {
                                        "tlr/teacher_real_rollouts": float(_n_real),
                                        "tlr/actual_teacher_queries": float(_n_real),
                                        "tlr/verifier_queries": 0.0,
                                        "tlr/verifier_seconds": 0.0,
                                        "tlr/teacher_dummy_rollouts": float(_n_dummy),
                                        "tlr/teacher_padded_batch_size": float(_n_padded),
                                        "tlr/teacher_input_tokens": float(input_tokens),
                                        "tlr/teacher_scored_tokens": float(scored_tokens),
                                        "tlr/teacher_forward_seconds": teacher_call_latency_sec,
                                        "tlr/teacher_gpu_hours": teacher_hours,
                                        "tlr/selected_sampled_reverse_kl_mean": (
                                            selected_reverse_kl_mean
                                        ),
                                    }
                                )

                                selected_to_row = {
                                    int(original): row
                                    for row, original in enumerate(selected_indices.tolist())
                                }
                                for original_index, record in enumerate(tlr_opd_context["records"]):
                                    if original_index not in selected_to_row:
                                        continue
                                    row = selected_to_row[original_index]
                                    row_mask = response_mask_tlr[row]
                                    row_logratio = sampled_logratio[row][row_mask]
                                    record.update(
                                        {
                                            "optimizer_step": self.global_steps,
                                            "teacher_input_tokens": int(input_tokens_by_row[row]),
                                            "teacher_scored_tokens": int(scored_tokens_by_row[row]),
                                            "sampled_logratio_mean": float(row_logratio.mean()),
                                            "abs_sampled_logratio_mean": float(
                                                row_logratio.abs().mean()
                                            ),
                                            "sampled_reverse_kl_sum": float(row_logratio.sum()),
                                            "sampled_reverse_kl_token_mean": float(
                                                row_logratio.mean()
                                            ),
                                        }
                                    )
                                self._tlr_csv.append(tlr_opd_context["records"])

                            # Emit teacher routing metrics.
                            metrics.update({
                                "teacher_routing/candidates_before_filter": float(_teacher_filter_stats.n_before),
                                "teacher_routing/drop_overlength": float(_teacher_filter_stats.drop_overlength),
                                "teacher_routing/drop_truncated": float(_teacher_filter_stats.drop_truncated),
                                "teacher_routing/truncation_rate": float(
                                    _teacher_filter_stats.drop_truncated
                                ) / max(_teacher_filter_stats.n_before, 1),
                                "teacher_routing/drop_invalid": float(_teacher_filter_stats.drop_invalid),
                                "teacher_routing/real_rollouts": float(_n_real),
                                "teacher_routing/dummy_rollouts": float(_n_dummy),
                                "teacher_routing/padded_batch_size": float(_n_padded),
                                "teacher_routing/teacher_dp": float(_teacher_dp),
                                # Invariant: no valid candidate was dropped for DP alignment.
                                "teacher_routing/drop_dp_alignment": 0.0,
                            })
                            # Update FF-OPD metrics with dummy padding info.
                            if self._ff_opd_enabled and ff_opd_context is not None:
                                ff_opd_context.metrics["ff/teacher_dummy_rollouts"] = float(_n_dummy)
                                ff_opd_context.metrics["ff/teacher_padded_batch_size"] = float(_n_padded)
                                ff_opd_context.metrics["ff/teacher_real_rollouts"] = float(_n_real)
                                ff_opd_context.metrics["ff/teacher_drop_dp_alignment"] = 0.0
                                ff_opd_context.metrics["ff/post_selection_truncated_drop"] = float(
                                    _teacher_filter_stats.drop_truncated
                                )
                                for record in ff_opd_context.records:
                                    if record["selected_for_teacher"]:
                                        record["teacher_drop_truncated_after_selection"] = _teacher_filter_stats.drop_truncated
                                        record["teacher_real_rollout_count"] = _n_real
                                        record["teacher_dummy_rollout_count"] = _n_dummy
                                _logger.info(
                                    "[FF teacher] frontier_candidates=%d selected_by_budget=%d "
                                    "teacher_real_rollouts=%d teacher_dummy_rollouts=%d "
                                    "post_selection_truncated_drop=%d no_success_selected=0 all_correct_selected=0",
                                    int(ff_opd_context.metrics["ff/frontier_candidates"]),
                                    int(ff_opd_context.metrics["ff/frontier_selected"]),
                                    _n_real,
                                    _n_dummy,
                                    _teacher_filter_stats.drop_truncated,
                                )
                            # The critical invariant: no valid candidate was silently discarded.
                            assert _n_padded == _n_real + _n_dummy, (
                                f"Padded batch size {_n_padded} != real {_n_real} + dummy {_n_dummy}"
                            )

                            if top_k > 0:
                                # Compute Student-on-Teacher cross-support scores.
                                # They remain auxiliary for TA diagnostics;
                                # every method keeps the Teacher worker's 2D
                                # sampled-token reward as the Actor target.
                                with marked_timer("compute_distillation_reward", timing_raw, color="orange"):
                                    # After strip_dummy_samples the real batch size may not be
                                    # divisible by the actor DP world size.  Pad to the next
                                    # multiple, dispatch, then unpad the result before union.
                                    _actor_dp = int(self.actor_rollout_wg.world_size)
                                    batch_for_distill, _distill_pad_size = pad_dataproto_to_divisor(
                                        batch, size_divisor=_actor_dp
                                    )
                                    distillation_output_padded = self.actor_rollout_wg.compute_distillation_reward(
                                        batch_for_distill
                                    )
                                    distillation_output = unpad_dataproto(
                                        distillation_output_padded, pad_size=_distill_pad_size
                                    )
                                    batch = batch.union(distillation_output)
                                if self._ff_opd_enabled:
                                    self._finalize_ff_opd_records(
                                        batch, ff_opd_context, teacher_call_latency_sec
                                    )
                                    # Only count real (non-dummy) tokens for FF metrics.
                                    supervised = float(batch.batch["response_mask"].sum().item())
                                    processing = float(batch.batch["attention_mask"].sum().item())
                                    metrics.update(
                                        {
                                            "ff/total_teacher_processing_tokens": processing,
                                            "ff/teacher_scored_tokens": supervised,
                                            "ff/teacher_supervised_tokens": supervised,
                                            "ff/sampled_reverse_kl_token_mean": float(
                                                -batch.batch["sampled_opd_rm_scores"][batch.batch["response_mask"].bool()]
                                                .mean()
                                                .item()
                                            ),
                                            "ff/student_sampled_logprob_mean": float(
                                                batch.batch["old_log_probs"][batch.batch["response_mask"].bool()]
                                                .mean()
                                                .item()
                                            ),
                                            "ff/teacher_sampled_logprob_mean": float(
                                                (
                                                    batch.batch["old_log_probs"]
                                                    + batch.batch["sampled_opd_rm_scores"]
                                                )[batch.batch["response_mask"].bool()]
                                                .mean()
                                                .item()
                                            ),
                                        }
                                    )
                            elif self._ff_opd_enabled:
                                self._finalize_ff_opd_records(batch, ff_opd_context, teacher_call_latency_sec)
                        
                        # Plot overlapping tokens for Reverse KL
                        if (
                            self.global_steps == 1 or self.global_steps % 10 == 0
                        ) and "student_valid_counts" in batch.batch.keys():
                            try:
                                import matplotlib.pyplot as plt
                                try:
                                    import swanlab as _swanlab
                                except ImportError:
                                    _swanlab = None

                                response_mask = batch.batch["response_mask"]
                                valid_denom = response_mask.sum(dim=0) + 1e-6

                                plot_data = {}
                                
                                # Calculate Student Candidates
                                if "student_valid_counts" in batch.batch.keys():
                                    student_counts = batch.batch["student_valid_counts"].float()
                                    avg_student_counts = (student_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Student"] = avg_student_counts.detach().cpu().numpy()
                                
                                # Calculate Teacher and Overlap Candidates if available
                                if "teacher_valid_counts" in batch.batch.keys():
                                    teacher_counts = batch.batch["teacher_valid_counts"].float()
                                    avg_teacher_counts = (teacher_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Teacher"] = avg_teacher_counts.detach().cpu().numpy()
                                    
                                if "overlap_mask" in batch.batch.keys():
                                    # overlap_mask is (BS, SeqLen, K), sum over K to get counts
                                    overlap_mask = batch.batch["overlap_mask"].float()
                                    overlap_counts = overlap_mask.sum(dim=-1)  # (BS, SeqLen)
                                    avg_overlap_counts = (overlap_counts * response_mask).sum(dim=0) / valid_denom
                                    plot_data["Overlap"] = avg_overlap_counts.detach().cpu().numpy()
                                
                                # Plot 1: Candidate Counts
                                plt.figure(figsize=(10, 6))
                                for label, data in plot_data.items():
                                    mean_val = data.mean()
                                    plt.plot(data, label=f"Avg {label} (mean: {mean_val:.2f})")
                                
                                plt.title(f"Avg Candidate Tokens per Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Candidate Count")
                                plt.legend()
                                plt.grid(True)
                                plt.tight_layout()
                                
                                plt.close()

                                if _swanlab is not None:
                                    count_plot = _swanlab.Image(plt, caption=f"Candidate Counts (Step {self.global_steps})")

                                    # Plot 2: Ratios
                                    log_payload = {"viz/candidate_counts": count_plot}

                                    if "Overlap" in plot_data and "Student" in plot_data and "Teacher" in plot_data:
                                        ratio_student = plot_data["Overlap"] / (plot_data["Student"] + 1e-6)
                                        ratio_teacher = plot_data["Overlap"] / (plot_data["Teacher"] + 1e-6)

                                        # Plot 2a: Overlap / Student
                                        plt.figure(figsize=(10, 6))
                                        plt.plot(ratio_student, label=f"Overlap / Student (mean: {ratio_student.mean():.2f})", color='tab:blue')
                                        plt.title(f"Overlap / Student Ratio (Step {self.global_steps})")
                                        plt.xlabel("Position")
                                        plt.ylabel("Ratio")
                                        plt.ylim(-0.05, 1.05)
                                        plt.legend()
                                        plt.grid(True)
                                        plt.tight_layout()

                                        ratio_student_plot = _swanlab.Image(plt, caption=f"Overlap / Student Ratio (Step {self.global_steps})")
                                        plt.close()
                                        log_payload["viz/overlap_ratio_student"] = ratio_student_plot

                                        # Plot 2b: Overlap / Teacher
                                        plt.figure(figsize=(10, 6))
                                        plt.plot(ratio_teacher, label=f"Overlap / Teacher (mean: {ratio_teacher.mean():.2f})", color='tab:orange')
                                        plt.title(f"Overlap / Teacher Ratio (Step {self.global_steps})")
                                        plt.xlabel("Position")
                                        plt.ylabel("Ratio")
                                        plt.ylim(-0.05, 1.05)
                                        plt.legend()
                                        plt.grid(True)
                                        plt.tight_layout()

                                        ratio_teacher_plot = _swanlab.Image(plt, caption=f"Overlap / Teacher Ratio (Step {self.global_steps})")
                                        plt.close()
                                        log_payload["viz/overlap_ratio_teacher"] = ratio_teacher_plot

                                    logger.log(log_payload, step=self.global_steps)
                                    print(f"Logged candidate plots to SwanLab at step {self.global_steps}")
                                else:
                                    _logger.debug(
                                        "[viz] swanlab not installed; skipping candidate count plots at step %d.",
                                        self.global_steps,
                                    )
                                
                            except Exception as e:
                                print(f"Error plotting candidate counts: {e}")
                        
                        
                        # Keep student_top_k_log_probs for potential use in policy loss computation
                        # Only pop temporary visualization data
                        if "student_valid_counts" in batch.batch.keys():
                             batch.batch.pop("student_valid_counts")
                        if "teacher_valid_counts" in batch.batch.keys():
                             batch.batch.pop("teacher_valid_counts")
                        if "overlap_counts" in batch.batch.keys():
                             batch.batch.pop("overlap_counts")


                        if self._tlr_opd_enabled:
                            # Ordinary TLR needs no answer verifier. The Teacher
                            # sampled-token reward is the complete OPSD target;
                            # calling reward_fn here would grade correctness only
                            # to discard that result in favor of rm_scores.
                            reward_tensor = batch.batch.get("rm_scores")
                            if reward_tensor is None:
                                raise RuntimeError("TLR requires rm_scores from the Teacher forward")
                            reward_extra_infos_dict = {}
                        elif self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            if "format_mask" in reward_extra_infos_dict.keys():
                                batch.batch["format_mask"] = reward_extra_infos_dict["format_mask"]
                    
                    from verl.trainer.ppo.rollout_corr_helper import (
                        compute_rollout_correction_and_add_to_batch,
                        maybe_apply_rollout_correction,
                    )

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    need_recomputation = maybe_apply_rollout_correction(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                    if need_recomputation:
                        # Optimization: Reuse data if available from Distillation Phase
                        entropys = None
                        if "old_log_probs" in batch.batch.keys() and "entropys" in batch.batch.keys():
                             entropys = batch.batch["entropys"]
                             print("We don't need to re-merge old_log_probs, it's already there.")

                        else:
                             # Legacy Path: Must recompute if not present
                             with marked_timer("old_log_prob", timing_raw, color="blue"):
                                 old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                 entropys = old_log_prob.batch["entropys"]
                                 batch = batch.union(old_log_prob)
                                 
                                 # Remove top-k keys from old_log_prob if they already exist in batch
                                 # (they may have been modified for union strategy)
                                 for key in ["student_top_k_ids", "student_top_k_log_probs"]:
                                     if key in batch.batch.keys() and key in old_log_prob.batch.keys():
                                         pass # Already handled by union? Warning: Union might overwrite if not careful.
                                         # The original code had a manual check here, but batch.union generally overwrites.
                                         # Assuming Actor's new log prob output is the "source of truth" if we recompute.

                        if entropys is not None:
                            response_masks = batch.batch["response_mask"]
                            if "format_mask" in batch.batch.keys():
                                response_masks = response_masks * batch.batch["format_mask"].unsqueeze(-1)
                            
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            metrics.update({"actor/entropy": entropy_agg.detach().item()})

                            # Compute teacher entropy metric if available
                            if "teacher_entropy" in batch.batch.keys():
                                teacher_entropy = batch.batch["teacher_entropy"]
                                teacher_entropy_agg = agg_loss(
                                    loss_mat=teacher_entropy, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                metrics.update({"teacher/entropy": teacher_entropy_agg.detach().item()})

                            # Cleanup: We are done with entropys
                            if "entropys" in batch.batch.keys():
                                batch.batch.pop("entropys")


                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async and not self._tlr_opd_enabled:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if "true_reward_score" in reward_extra_infos_dict:
                            true_reward_val = reward_extra_infos_dict["true_reward_score"]
                            if isinstance(true_reward_val, torch.Tensor):
                                batch.batch["true_reward_score"] = true_reward_val
                            else:
                                batch.batch["true_reward_score"] = torch.as_tensor(
                                    true_reward_val,
                                    device=reward_tensor.device,
                                    dtype=reward_tensor.dtype,
                                )
                        else:
                            batch.batch["true_reward_score"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction weights centrally (once per batch)
                        # This corrects for off-policy issues (policy mismatch, model staleness, etc.)
                        # Also computes off-policy diagnostic metrics (KL, PPL, etc.)
                        if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        if gradient_diagnostics_enabled:
                            diagnostic_tensors = {key: batch.batch[key] for key in ROLLOUT_DIAGNOSTIC_KEYS}
                            current_lengths = batch.batch["response_mask"].sum(dim=-1).float()
                            if not torch.equal(
                                current_lengths.cpu(), diagnostic_tensors["rollout_response_length"].float().cpu()
                            ):
                                raise RuntimeError("Full OPD response mask changed after gradient diagnostics")
                            if "true_reward_score" not in batch.batch:
                                raise RuntimeError(
                                    "rollout gradient diagnostics require per-rollout true_reward_score "
                                    "from the rule-based verifier on the Full OPD path"
                                )
                            true_reward_score = batch.batch["true_reward_score"].detach().float()
                            if true_reward_score.dim() > 1:
                                # The rule-based verifier writes each score at the last valid
                                # token of its response ([batch, response_len] with a single
                                # non-zero entry per row); reduce to one score per rollout.
                                true_reward_score = true_reward_score.amax(dim=-1)
                            true_rewards = true_reward_score.cpu().numpy()
                            if "format_mask" in batch.batch:
                                format_mask = batch.batch["format_mask"].float()
                                diagnostic_tensors = {
                                    key: value * format_mask.to(value.device)
                                    for key, value in diagnostic_tensors.items()
                                }
                            gradient_metrics, gradient_records = summarize_rollout_gradient_diagnostics(
                                uids=batch.non_tensor_batch["uid"],
                                rollout_tensors=diagnostic_tensors,
                                expected_group_size=int(self.config.actor_rollout_ref.rollout.n),
                                eps=float(gradient_diagnostics_config.eps),
                                true_rewards=true_rewards,
                            )
                            metrics.update(gradient_metrics)
                            if gradient_metrics["grad_concentration/incomplete_group_count"] or gradient_metrics[
                                "grad_concentration/overfull_group_count"
                            ]:
                                _logger.warning(
                                    "Rollout gradient diagnostics skipped malformed UID groups: "
                                    "incomplete=%d overfull=%d",
                                    int(gradient_metrics["grad_concentration/incomplete_group_count"]),
                                    int(gradient_metrics["grad_concentration/overfull_group_count"]),
                                )
                            if gradient_diagnostics_config.save_per_rollout:
                                output_path = Path(gradient_diagnostics_config.save_dir) / (
                                    "rollout_gradient_diagnostics.jsonl"
                                )
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                with output_path.open("a", encoding="utf-8") as stream:
                                    for record in gradient_records:
                                        record["global_step"] = self.global_steps
                                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            batch.pop(batch_keys=list(ROLLOUT_DIAGNOSTIC_KEYS))


                        # ============================================================
                        # TA-OPD: Compute token selection mask after teacher scoring
                        # ============================================================
                        ta_opd_enable = self.config.actor_rollout_ref.rollout.get("ta_opd_enable", False)
                        if ta_opd_enable and batch.batch["advantages"].dim() != 2:
                            raise RuntimeError(
                                "TA-OPD must use sampled-token [B, T] advantages; got shape "
                                f"{tuple(batch.batch['advantages'].shape)}. Top-k tensors are selector-only."
                            )
                        cost_metrics_enable = self.config.actor_rollout_ref.rollout.get("cost_metrics_enable", True)

                        # Initialize cost meter if not already done
                        if not hasattr(self, "_cost_meter"):
                            self._cost_meter = CostMeter(
                                enabled=cost_metrics_enable,
                                track_timing=self.config.actor_rollout_ref.rollout.get("cost_metrics_track_timing", True),
                                track_gpu_memory=self.config.actor_rollout_ref.rollout.get("cost_metrics_track_gpu_memory", True),
                                track_teacher_call=self.config.actor_rollout_ref.rollout.get("cost_metrics_track_teacher_call", True),
                                warn_once=self.config.actor_rollout_ref.rollout.get("cost_metrics_warn_once", True),
                            )
                        self._cost_meter.reset_step()

                        # Full-query token counts.
                        response_mask = batch.batch["response_mask"]
                        prompt_mask = batch.batch["attention_mask"][:, :-response_mask.shape[-1]]
                        prompt_token_count = int(prompt_mask.sum().item())
                        response_token_count = int(response_mask.sum().item())
                        valid_response_token_count = int(response_mask.sum().item())
                        full_valid_response_token_count = valid_response_token_count
                        full_teacher_processed_token_count = prompt_token_count + response_token_count

                        if teacher_call_latency_sec is not None:
                            actual_teacher_logprob_count = 0
                            configured_topk = self.config.actor_rollout_ref.rollout.get(
                                "log_prob_top_k", 0
                            )
                            returned_topk = (
                                self.config.actor_rollout_ref.rollout.get("ta_opd_topk", 16)
                                if ta_opd_enable
                                else configured_topk
                            )
                            teacher_logprob_tensor = batch.batch.get(
                                "teacher_top_k_log_probs", None
                            )
                            if teacher_logprob_tensor is None:
                                teacher_logprob_tensor = batch.batch.get(
                                    "teacher_on_student_log_probs", None
                                )
                            if returned_topk <= 0:
                                teacher_logprob_tensor = batch.batch.get("rm_scores", None)
                            if teacher_logprob_tensor is not None:
                                actual_teacher_logprob_count = int(
                                    teacher_logprob_tensor[response_mask.bool()].numel()
                                )
                            # add_teacher_call owns call metadata only. Token accounting
                            # is done once below by add_tokens to avoid double counting.
                            self._cost_meter.add_teacher_call(
                                call_count=1,
                                call_type="reward_model_forward",
                                sequence_count=batch.batch["responses"].shape[0],
                                batch_size=batch.batch["responses"].shape[0],
                                returned_logprob_count=actual_teacher_logprob_count,
                                expected_logprobs_per_scored_token=max(
                                    returned_topk, 1
                                ),
                                latency_sec=teacher_call_latency_sec,
                            )

                        # Compute TA-OPD mask if enabled
                        ta_opd_selected_mask = None
                        ta_opd_stats = {}
                        ta_opd_details = None
                        student_topk_log_probs = None
                        teacher_topk_log_probs = None
                        if ta_opd_enable:
                            top_k_val = self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0)
                            if top_k_val <= 0:
                                top_k_val = self.config.actor_rollout_ref.rollout.get("ta_opd_topk", 16)

                            student_topk_log_probs = batch.batch.get("student_top_k_log_probs", None)
                            student_topk_indices = batch.batch.get("student_top_k_ids", None)
                            teacher_topk_log_probs = batch.batch.get("teacher_top_k_log_probs", None)
                            teacher_topk_indices = batch.batch.get("teacher_top_k_ids", None)
                            teacher_on_student_log_probs = batch.batch.get("teacher_on_student_log_probs", None)
                            student_on_teacher_log_probs = batch.batch.get("student_log_probs_on_teacher_ids", None)

                            ta_inputs = {
                                "student_top_k_log_probs": student_topk_log_probs,
                                "student_top_k_ids": student_topk_indices,
                                "teacher_top_k_log_probs": teacher_topk_log_probs,
                                "teacher_top_k_ids": teacher_topk_indices,
                                "teacher_on_student_log_probs": teacher_on_student_log_probs,
                                "student_log_probs_on_teacher_ids": student_on_teacher_log_probs,
                            }
                            missing_ta_inputs = [name for name, value in ta_inputs.items() if value is None]
                            if not missing_ta_inputs:
                                retain_token_count = self.config.actor_rollout_ref.rollout.get(
                                    "ta_opd_retain_token_count", None
                                )
                                ta_opd_selected_mask, _, ta_opd_stats = compute_ta_opd_mask(
                                    student_topk_log_probs=student_topk_log_probs,
                                    student_topk_indices=student_topk_indices,
                                    teacher_topk_log_probs=teacher_topk_log_probs,
                                    teacher_topk_indices=teacher_topk_indices,
                                    teacher_on_student_log_probs=teacher_on_student_log_probs,
                                    student_on_teacher_log_probs=student_on_teacher_log_probs,
                                    response_mask=response_mask,
                                    topk=self.config.actor_rollout_ref.rollout.get("ta_opd_topk", 16),
                                    retain_ratio=self.config.actor_rollout_ref.rollout.get(
                                        "ta_opd_retain_ratio", 0.10
                                    ),
                                    retain_token_count=retain_token_count,
                                    normalize_scope=self.config.actor_rollout_ref.rollout.get(
                                        "ta_opd_normalize_scope", "batch"
                                    ),
                                    mode=self.config.actor_rollout_ref.rollout.get(
                                        "ta_opd_mode", "teachability"
                                    ),
                                    detach_score=self.config.actor_rollout_ref.rollout.get(
                                        "ta_opd_detach_score", True
                                    ),
                                )
                                metrics.update(ta_opd_stats)

                                # Store selected_mask in batch for actor to use
                                batch.batch["ta_opd_selected_mask"] = ta_opd_selected_mask
                                # Scalar configuration belongs in meta_info so DataProto
                                # sharding does not treat it as a batch tensor.
                                batch.meta_info["ta_opd_normalize_by"] = self.config.actor_rollout_ref.rollout.get(
                                    "ta_opd_normalize_by", "selected_tokens"
                                )

                                supervised_count = int(ta_opd_selected_mask.sum().item())
                            else:
                                raise RuntimeError(
                                    "TA-OPD requires exact cross-support log-probabilities on the Student/Teacher "
                                    f"top-k union; missing tensors: {missing_ta_inputs}. Use top_k_strategy=union."
                                )
                        else:
                            supervised_count = valid_response_token_count

                        # Track token statistics
                        teacher_processed_tokens = prompt_token_count + response_token_count
                        teacher_scored_tokens = valid_response_token_count
                        self._cost_meter.add_tokens(
                            prompt_token_count=prompt_token_count,
                            response_token_count=response_token_count,
                            valid_response_token_count=full_valid_response_token_count,
                            supervised_token_count=supervised_count,
                            teacher_processed_input_token_count=teacher_processed_tokens,
                            teacher_scored_response_token_count=teacher_scored_tokens,
                            full_query_teacher_processed_input_token_count=full_teacher_processed_token_count,
                            full_query_teacher_scored_response_token_count=full_valid_response_token_count,
                            full_query_teacher_call_count=1,
                        )

                        metrics["opd_query/full_valid_response_tokens"] = (
                            full_valid_response_token_count
                        )

                        # Compute OPD metrics
                        advantages_for_loss = batch.batch["advantages"]
                        if advantages_for_loss.dim() == 3:
                            pt_loss = advantages_for_loss.sum(dim=-1)
                        else:
                            pt_loss = advantages_for_loss
                        opd_metrics = compute_opd_metrics(
                            response_mask=response_mask,
                            selected_mask=(
                                ta_opd_selected_mask
                                if ta_opd_selected_mask is not None
                                else batch.batch.get("ta_opd_selected_mask", None)
                            ),
                            prompt_token_count=prompt_token_count,
                            teacher_scored_response_token_count=valid_response_token_count,
                            teacher_processed_input_token_count=prompt_token_count + response_token_count,
                            per_token_loss=pt_loss,
                            student_topk_log_probs=student_topk_log_probs if ta_opd_enable else None,
                            teacher_topk_log_probs=teacher_topk_log_probs if ta_opd_enable else None,
                            ta_opd_enabled=ta_opd_enable,
                            full_query_teacher_scored=full_valid_response_token_count,
                            full_query_teacher_processed=full_teacher_processed_token_count,
                            full_valid_response_token_count=full_valid_response_token_count,
                        )
                        metrics.update(opd_metrics)

                        # Collect cost metrics
                        cost_metrics = self._cost_meter.get_step_metrics()
                        metrics.update(cost_metrics)

                        # --- Top-K Metrics Analysis (Chunked) ---
                        if "overlap_mask" in batch.batch.keys() and "advantages" in batch.batch.keys():
                            try:
                                overlap_mask = batch.batch["overlap_mask"].float() # (BS, SeqLen, K)
                                advantages = batch.batch["advantages"] # (BS, SeqLen, K) or (BS, SeqLen, 2K) for union
                                
                                response_mask = batch.batch["response_mask"] # (BS, SeqLen)
                                max_len = response_mask.shape[-1]
                                top_k = batch.meta_info.get("log_prob_top_k", 0)
                                strategy = batch.meta_info.get("top_k_strategy", "only_stu")
                                
                                # For union strategy, get teacher_in_student mask
                                teacher_in_student_mask = batch.batch.get("teacher_in_student_mask", None) # (BS, SeqLen, K) or None
                                
                                # Get log probs for p_sum metrics (student and teacher probabilities)
                                student_log_probs = batch.batch.get("student_top_k_log_probs", None)  # (BS, SeqLen, K)
                                teacher_on_stu_log_probs = batch.batch.get("teacher_on_student_log_probs", None)  # (BS, SeqLen, K)
                                teacher_log_probs = batch.batch.get("teacher_top_k_log_probs", None)  # (BS, SeqLen, K)
                                student_on_tch_log_probs = batch.batch.get("student_log_probs_on_teacher_ids", None)  # (BS, SeqLen, K)

                                if top_k > 0 and advantages.dim() == 3:
                                    adv_k = advantages.shape[-1]  # K or 2K
                                    is_union = (strategy == "union" or strategy == "union-intersection") and (adv_k == 2 * top_k)
                                    
                                    # --- Global Metrics ---
                                    # Expand response mask to match advantages shape
                                    global_valid_mask_float = response_mask.unsqueeze(-1).expand(advantages.shape[0], advantages.shape[1], adv_k).float()
                                    global_valid_mask_bool = global_valid_mask_float > 0.5
                                    
                                    if is_union:
                                        # For union: front K is student, back K is teacher
                                        # overlap_mask: (B, T, K) - student id in teacher top k
                                        # teacher_in_student_mask: (B, T, K) - teacher id in student top k
                                        
                                        student_overlap = overlap_mask # (B, T, K)
                                        teacher_overlap = teacher_in_student_mask if teacher_in_student_mask is not None else torch.zeros_like(overlap_mask)
                                        
                                        # Build masks for the full 2K dimension
                                        # Front K: student top k
                                        #   - intersection: student_overlap > 0.5
                                        #   - only_stu: student_overlap < 0.5
                                        # Back K: teacher top k (only valid if not duplicate, i.e., ~teacher_overlap)
                                        #   - only_tch: ~teacher_overlap (teacher not in student)
                                        #   - (intersection on teacher side would be duplicate, already masked)
                                        
                                        student_adv = advantages[:, :, :top_k] # (B, T, K)
                                        teacher_adv = advantages[:, :, top_k:] # (B, T, K)
                                        
                                        student_valid = response_mask.unsqueeze(-1).expand_as(student_overlap).bool()
                                        teacher_valid = response_mask.unsqueeze(-1).expand_as(teacher_overlap).bool()
                                        
                                        # 1. Global Overlap Ratio (student side only for consistency)
                                        total_valid_k = student_valid.float().sum()
                                        total_overlap_k = (student_overlap * student_valid.float()).sum()
                                        
                                        if total_valid_k > 0:
                                            metrics["val-topk/overlap_ratio"] = (total_overlap_k / total_valid_k).item()
                                        
                                        # 2. Intersection Advantage (student tokens in teacher top k)
                                        mask_inter = (student_overlap > 0.5) & student_valid
                                        if mask_inter.any():
                                            avg_adv_inter = student_adv[mask_inter].mean()
                                            metrics["val-topk/adv_intersection"] = avg_adv_inter.item()
                                            
                                            # Compute p metrics for intersection (union strategy)
                                            if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                student_p = torch.exp(student_log_probs)  # (B, T, K)
                                                teacher_p = torch.exp(teacher_on_stu_log_probs)  # (B, T, K)
                                                inter_positions = mask_inter.any(dim=-1)  # (B, T)
                                                
                                                # p_sum metrics
                                                student_p_masked = torch.where(mask_inter, student_p, torch.zeros_like(student_p))
                                                teacher_p_masked = torch.where(mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                student_p_sum = student_p_masked.sum(dim=-1)
                                                teacher_p_sum = teacher_p_masked.sum(dim=-1)
                                                metrics["val-topk/student_p_sum_intersection"] = student_p_sum[inter_positions].mean().item()
                                                metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_sum[inter_positions].mean().item()
                                                
                                                # max_p metrics
                                                student_p_for_max = torch.where(mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                teacher_p_for_max = torch.where(mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                
                                                max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                adv_at_max_stu = student_adv.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                adv_at_max_tch = student_adv.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                
                                                # max/min adv metrics
                                                adv_for_max = torch.where(mask_inter, student_adv, torch.full_like(student_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_inter, student_adv, torch.full_like(student_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = student_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = student_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                        
                                        # 3. Only Student Advantage (student tokens NOT in teacher top k)
                                        mask_only_stu = (student_overlap < 0.5) & student_valid
                                        if mask_only_stu.any():
                                            avg_adv_only_stu = student_adv[mask_only_stu].mean()
                                            metrics["val-topk/adv_only_stu"] = avg_adv_only_stu.item()
                                            
                                            # val-extrema metrics for only_stu
                                            if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                only_stu_positions = mask_only_stu.any(dim=-1)
                                                adv_for_max = torch.where(mask_only_stu, student_adv, torch.full_like(student_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_only_stu, student_adv, torch.full_like(student_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = student_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = student_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_only_stu"] = max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_only_stu"] = stu_p_at_max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_only_stu"] = tch_p_at_max_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/min_adv_only_stu"] = min_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_only_stu"] = stu_p_at_min_adv[only_stu_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_only_stu"] = tch_p_at_min_adv[only_stu_positions].mean().item()
                                        
                                        # 4. Only Teacher Advantage (teacher tokens NOT in student top k)
                                        # These are the valid teacher tokens (not duplicated)
                                        mask_only_tch = (teacher_overlap < 0.5) & teacher_valid
                                        if mask_only_tch.any():
                                            avg_adv_only_tch = teacher_adv[mask_only_tch].mean()
                                            metrics["val-topk/adv_only_tch"] = avg_adv_only_tch.item()
                                            
                                            # val-extrema metrics for only_tch
                                            # For teacher tokens, we use teacher_adv and corresponding probabilities
                                            if teacher_log_probs is not None and student_on_tch_log_probs is not None:
                                                only_tch_positions = mask_only_tch.any(dim=-1)
                                                teacher_p_tch = torch.exp(teacher_log_probs)  # (B, T, K)
                                                student_p_tch = torch.exp(student_on_tch_log_probs)  # (B, T, K)
                                                
                                                adv_for_max = torch.where(mask_only_tch, teacher_adv, torch.full_like(teacher_adv, float('-inf')))
                                                adv_for_min = torch.where(mask_only_tch, teacher_adv, torch.full_like(teacher_adv, float('inf')))
                                                max_adv_idx = adv_for_max.argmax(dim=-1)
                                                min_adv_idx = adv_for_min.argmin(dim=-1)
                                                
                                                max_adv = teacher_adv.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_max_adv = student_p_tch.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_max_adv = teacher_p_tch.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                min_adv = teacher_adv.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                stu_p_at_min_adv = student_p_tch.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                tch_p_at_min_adv = teacher_p_tch.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                
                                                metrics["val-extrema/max_adv_only_tch"] = max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/student_p_at_max_adv_only_tch"] = stu_p_at_max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_max_adv_only_tch"] = tch_p_at_max_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/min_adv_only_tch"] = min_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/student_p_at_min_adv_only_tch"] = stu_p_at_min_adv[only_tch_positions].mean().item()
                                                metrics["val-extrema/teacher_p_at_min_adv_only_tch"] = tch_p_at_min_adv[only_tch_positions].mean().item()
                                        
                                        # --- Chunk-level metrics for union ---
                                        chunk_size = 1024
                                        for start_idx in range(0, max_len, chunk_size):
                                            end_idx = min(start_idx + chunk_size, max_len)
                                            chunk_key = f"{start_idx}_{end_idx}"
                                            
                                            chunk_response_mask = response_mask[:, start_idx:end_idx].bool()
                                            chunk_student_overlap = student_overlap[:, start_idx:end_idx]
                                            chunk_teacher_overlap = teacher_overlap[:, start_idx:end_idx]
                                            chunk_student_adv = student_adv[:, start_idx:end_idx]
                                            chunk_teacher_adv = teacher_adv[:, start_idx:end_idx]
                                            
                                            if not chunk_response_mask.any():
                                                continue
                                            
                                            chunk_student_valid = chunk_response_mask.unsqueeze(-1).expand_as(chunk_student_overlap)
                                            chunk_teacher_valid = chunk_response_mask.unsqueeze(-1).expand_as(chunk_teacher_overlap)
                                            
                                            # Overlap Ratio
                                            total_valid = chunk_student_valid.float().sum()
                                            total_overlap = (chunk_student_overlap * chunk_student_valid.float()).sum()
                                            if total_valid > 0:
                                                metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap / total_valid).item()
                                            
                                            # Intersection
                                            mask_inter_c = (chunk_student_overlap > 0.5) & chunk_student_valid
                                            if mask_inter_c.any():
                                                metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = chunk_student_adv[mask_inter_c].mean().item()
                                                
                                                # Compute p metrics for intersection chunk (union strategy)
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    chunk_student_lp = student_log_probs[:, start_idx:end_idx]
                                                    chunk_teacher_lp = teacher_on_stu_log_probs[:, start_idx:end_idx]
                                                    student_p_c = torch.exp(chunk_student_lp)
                                                    teacher_p_c = torch.exp(chunk_teacher_lp)
                                                    inter_pos_c = mask_inter_c.any(dim=-1)
                                                    
                                                    # p_sum metrics
                                                    student_p_masked_c = torch.where(mask_inter_c, student_p_c, torch.zeros_like(student_p_c))
                                                    teacher_p_masked_c = torch.where(mask_inter_c, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                    metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                    
                                                    # max_p metrics
                                                    student_p_for_max_c = torch.where(mask_inter_c, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                    teacher_p_for_max_c = torch.where(mask_inter_c, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                    max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                    max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                    
                                                    max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu_c = chunk_student_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch_c = chunk_student_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                    metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                    
                                                    # max/min adv metrics
                                                    adv_for_max_c = torch.where(mask_inter_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_inter_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_student_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_student_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                            
                                            # Only Student
                                            mask_only_stu_c = (chunk_student_overlap < 0.5) & chunk_student_valid
                                            if mask_only_stu_c.any():
                                                metrics[f"val-topk/adv_only_stu_chunk_{chunk_key}"] = chunk_student_adv[mask_only_stu_c].mean().item()
                                                
                                                # val-extrema metrics for only_stu chunk
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    only_stu_pos_c = mask_only_stu_c.any(dim=-1)
                                                    adv_for_max_c = torch.where(mask_only_stu_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_only_stu_c, chunk_student_adv, torch.full_like(chunk_student_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_student_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_student_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_only_stu_chunk_{chunk_key}"] = max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_only_stu_chunk_{chunk_key}"] = stu_p_at_max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_only_stu_chunk_{chunk_key}"] = tch_p_at_max_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_only_stu_chunk_{chunk_key}"] = min_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_only_stu_chunk_{chunk_key}"] = stu_p_at_min_adv_c[only_stu_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_only_stu_chunk_{chunk_key}"] = tch_p_at_min_adv_c[only_stu_pos_c].mean().item()
                                            
                                            # Only Teacher
                                            mask_only_tch_c = (chunk_teacher_overlap < 0.5) & chunk_teacher_valid
                                            if mask_only_tch_c.any():
                                                metrics[f"val-topk/adv_only_tch_chunk_{chunk_key}"] = chunk_teacher_adv[mask_only_tch_c].mean().item()
                                                
                                                # val-extrema metrics for only_tch chunk
                                                if teacher_log_probs is not None and student_on_tch_log_probs is not None:
                                                    only_tch_pos_c = mask_only_tch_c.any(dim=-1)
                                                    chunk_teacher_lp = teacher_log_probs[:, start_idx:end_idx]
                                                    chunk_stu_on_tch_lp = student_on_tch_log_probs[:, start_idx:end_idx]
                                                    teacher_p_tch_c = torch.exp(chunk_teacher_lp)
                                                    student_p_tch_c = torch.exp(chunk_stu_on_tch_lp)
                                                    
                                                    adv_for_max_c = torch.where(mask_only_tch_c, chunk_teacher_adv, torch.full_like(chunk_teacher_adv, float('-inf')))
                                                    adv_for_min_c = torch.where(mask_only_tch_c, chunk_teacher_adv, torch.full_like(chunk_teacher_adv, float('inf')))
                                                    max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                    min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                    
                                                    max_adv_c = chunk_teacher_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv_c = student_p_tch_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv_c = teacher_p_tch_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    min_adv_c = chunk_teacher_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv_c = student_p_tch_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv_c = teacher_p_tch_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics[f"val-extrema/max_adv_only_tch_chunk_{chunk_key}"] = max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_max_adv_only_tch_chunk_{chunk_key}"] = stu_p_at_max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_max_adv_only_tch_chunk_{chunk_key}"] = tch_p_at_max_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/min_adv_only_tch_chunk_{chunk_key}"] = min_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/student_p_at_min_adv_only_tch_chunk_{chunk_key}"] = stu_p_at_min_adv_c[only_tch_pos_c].mean().item()
                                                    metrics[f"val-extrema/teacher_p_at_min_adv_only_tch_chunk_{chunk_key}"] = tch_p_at_min_adv_c[only_tch_pos_c].mean().item()
                                    
                                    else:
                                        # Non-union strategies (only_stu, only_tch, intersection)
                                        # For only_tch, use teacher_in_student_mask; for others, use overlap_mask
                                        if strategy == "only_tch" and "teacher_in_student_mask" in batch.batch:
                                            # For only_tch: advantages are for Teacher top k
                                            # teacher_in_student_mask: (B, T, K) - Teacher ID in Student top k
                                            tch_in_stu_mask = batch.batch["teacher_in_student_mask"]
                                            global_valid_mask_float_k = response_mask.unsqueeze(-1).expand_as(tch_in_stu_mask).float()
                                            global_valid_mask_bool_k = global_valid_mask_float_k > 0.5
                                            
                                            # 1. Global Overlap Ratio (Teacher side)
                                            global_total_valid_k = global_valid_mask_float_k.sum()
                                            global_total_overlap_k = (tch_in_stu_mask * global_valid_mask_float_k).sum()
                                            
                                            if global_total_valid_k > 0:
                                                metrics["val-topk/overlap_ratio"] = (global_total_overlap_k / global_total_valid_k).item()
                                            
                                            # 2. Intersection Advantage (Teacher tokens in Student top k)
                                            global_mask_inter = (tch_in_stu_mask > 0.5) & global_valid_mask_bool_k
                                            if global_mask_inter.any():
                                                global_avg_adv_inter = advantages[global_mask_inter].mean()
                                                metrics["val-topk/adv_intersection"] = global_avg_adv_inter.item()
                                                
                                                # Compute p metrics for intersection (only_tch strategy)
                                                student_on_tch_log_probs = batch.batch.get("student_log_probs_on_teacher_ids", None)
                                                teacher_top_k_lp = batch.batch.get("teacher_top_k_log_probs", None)
                                                if student_on_tch_log_probs is not None and teacher_top_k_lp is not None:
                                                    student_p = torch.exp(student_on_tch_log_probs)
                                                    teacher_p = torch.exp(teacher_top_k_lp)
                                                    inter_positions = global_mask_inter.any(dim=-1)
                                                    
                                                    # p_sum metrics
                                                    student_p_masked = torch.where(global_mask_inter, student_p, torch.zeros_like(student_p))
                                                    teacher_p_masked = torch.where(global_mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                    metrics["val-topk/student_p_sum_intersection"] = student_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    
                                                    # max_p metrics
                                                    student_p_for_max = torch.where(global_mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                    teacher_p_for_max = torch.where(global_mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                    max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                    max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                    
                                                    max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu = advantages.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch = advantages.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                    metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                    
                                                    # max/min adv metrics
                                                    adv_for_max = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('-inf')))
                                                    adv_for_min = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('inf')))
                                                    max_adv_idx = adv_for_max.argmax(dim=-1)
                                                    min_adv_idx = adv_for_min.argmin(dim=-1)
                                                    
                                                    max_adv = advantages.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    min_adv = advantages.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                                
                                            # 3. Only Teacher Advantage (Teacher tokens NOT in Student top k)
                                            global_mask_only_tch = (tch_in_stu_mask < 0.5) & global_valid_mask_bool_k
                                            if global_mask_only_tch.any():
                                                global_avg_adv_only_tch = advantages[global_mask_only_tch].mean()
                                                metrics["val-topk/adv_only_tch"] = global_avg_adv_only_tch.item()

                                            chunk_size = 1024
                                            for start_idx in range(0, max_len, chunk_size):
                                                end_idx = min(start_idx + chunk_size, max_len)
                                                chunk_key = f"{start_idx}_{end_idx}"
                                                
                                                chunk_response_mask = response_mask[:, start_idx:end_idx].bool()
                                                chunk_tch_in_stu = tch_in_stu_mask[:, start_idx:end_idx]
                                                chunk_adv = advantages[:, start_idx:end_idx]
                                                
                                                if not chunk_response_mask.any():
                                                    continue
                                                
                                                chunk_valid_mask = chunk_response_mask.unsqueeze(-1).expand_as(chunk_tch_in_stu)
                                                
                                                # Overlap Ratio
                                                total_valid_k = chunk_valid_mask.sum()
                                                total_overlap_k = (chunk_tch_in_stu * chunk_valid_mask.float()).sum()
                                                if total_valid_k > 0:
                                                    metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap_k / total_valid_k).item()
                                                
                                                # Intersection
                                                mask_inter = (chunk_tch_in_stu > 0.5) & chunk_valid_mask
                                                if mask_inter.any():
                                                    metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = chunk_adv[mask_inter].mean().item()
                                                    
                                                    # Compute p metrics for intersection chunk (only_tch strategy)
                                                    if student_on_tch_log_probs is not None and teacher_top_k_lp is not None:
                                                        chunk_stu_lp = student_on_tch_log_probs[:, start_idx:end_idx]
                                                        chunk_tch_lp = teacher_top_k_lp[:, start_idx:end_idx]
                                                        student_p_c = torch.exp(chunk_stu_lp)
                                                        teacher_p_c = torch.exp(chunk_tch_lp)
                                                        inter_pos_c = mask_inter.any(dim=-1)
                                                        
                                                        # p_sum metrics
                                                        student_p_masked_c = torch.where(mask_inter, student_p_c, torch.zeros_like(student_p_c))
                                                        teacher_p_masked_c = torch.where(mask_inter, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                        metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        
                                                        # max_p metrics
                                                        student_p_for_max_c = torch.where(mask_inter, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                        teacher_p_for_max_c = torch.where(mask_inter, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                        max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                        max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                        
                                                        max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_stu_c = chunk_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_tch_c = chunk_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                        
                                                        # max/min adv metrics
                                                        adv_for_max_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('-inf')))
                                                        adv_for_min_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('inf')))
                                                        max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                        min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                        
                                                        max_adv_c = chunk_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        min_adv_c = chunk_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                                
                                                # Only Teacher
                                                mask_only_tch = (chunk_tch_in_stu < 0.5) & chunk_valid_mask
                                                if mask_only_tch.any():
                                                    metrics[f"val-topk/adv_only_tch_chunk_{chunk_key}"] = chunk_adv[mask_only_tch].mean().item()
                                        else:
                                            # only_stu, intersection: overlap_mask and advantages both (B, T, K)
                                            global_valid_mask_float_k = response_mask.unsqueeze(-1).expand_as(overlap_mask).float()
                                            global_valid_mask_bool_k = global_valid_mask_float_k > 0.5
                                            
                                            # 1. Global Overlap Ratio
                                            global_total_valid_k = global_valid_mask_float_k.sum()
                                            global_total_overlap_k = (overlap_mask * global_valid_mask_float_k).sum()
                                            
                                            if global_total_valid_k > 0:
                                                metrics["val-topk/overlap_ratio"] = (global_total_overlap_k / global_total_valid_k).item()
                                            
                                            # 2. Global Advantage Analysis
                                            # Intersection Advantage
                                            global_mask_inter = (overlap_mask > 0.5) & global_valid_mask_bool_k
                                            if global_mask_inter.any():
                                                global_avg_adv_inter = advantages[global_mask_inter].mean()
                                                metrics["val-topk/adv_intersection"] = global_avg_adv_inter.item()
                                                
                                                # Compute p metrics for intersection (only_stu/intersection strategy)
                                                if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                    student_p = torch.exp(student_log_probs)
                                                    teacher_p = torch.exp(teacher_on_stu_log_probs)
                                                    inter_positions = global_mask_inter.any(dim=-1)
                                                    
                                                    # p_sum metrics
                                                    student_p_masked = torch.where(global_mask_inter, student_p, torch.zeros_like(student_p))
                                                    teacher_p_masked = torch.where(global_mask_inter, teacher_p, torch.zeros_like(teacher_p))
                                                    metrics["val-topk/student_p_sum_intersection"] = student_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_sum_intersection"] = teacher_p_masked.sum(dim=-1)[inter_positions].mean().item()
                                                    
                                                    # max_p metrics
                                                    student_p_for_max = torch.where(global_mask_inter, student_p, torch.full_like(student_p, float('-inf')))
                                                    teacher_p_for_max = torch.where(global_mask_inter, teacher_p, torch.full_like(teacher_p, float('-inf')))
                                                    max_stu_idx = student_p_for_max.argmax(dim=-1)
                                                    max_tch_idx = teacher_p_for_max.argmax(dim=-1)
                                                    
                                                    max_stu_p = student_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_stu = teacher_p.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_stu = advantages.gather(-1, max_stu_idx.unsqueeze(-1)).squeeze(-1)
                                                    max_tch_p = teacher_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_tch = student_p.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    adv_at_max_tch = advantages.gather(-1, max_tch_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-topk/max_student_p_intersection"] = max_stu_p[inter_positions].mean().item()
                                                    metrics["val-topk/teacher_p_at_max_student_intersection"] = tch_p_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_student_intersection"] = adv_at_max_stu[inter_positions].mean().item()
                                                    metrics["val-topk/max_teacher_p_intersection"] = max_tch_p[inter_positions].mean().item()
                                                    metrics["val-topk/student_p_at_max_teacher_intersection"] = stu_p_at_max_tch[inter_positions].mean().item()
                                                    metrics["val-topk/adv_at_max_teacher_intersection"] = adv_at_max_tch[inter_positions].mean().item()
                                                    
                                                    # max/min adv metrics
                                                    adv_for_max = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('-inf')))
                                                    adv_for_min = torch.where(global_mask_inter, advantages, torch.full_like(advantages, float('inf')))
                                                    max_adv_idx = adv_for_max.argmax(dim=-1)
                                                    min_adv_idx = adv_for_min.argmin(dim=-1)
                                                    
                                                    max_adv = advantages.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_max_adv = student_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_max_adv = teacher_p.gather(-1, max_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    min_adv = advantages.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    stu_p_at_min_adv = student_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    tch_p_at_min_adv = teacher_p.gather(-1, min_adv_idx.unsqueeze(-1)).squeeze(-1)
                                                    
                                                    metrics["val-extrema/max_adv_intersection"] = max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_max_adv_intersection"] = stu_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_max_adv_intersection"] = tch_p_at_max_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/min_adv_intersection"] = min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/student_p_at_min_adv_intersection"] = stu_p_at_min_adv[inter_positions].mean().item()
                                                    metrics["val-extrema/teacher_p_at_min_adv_intersection"] = tch_p_at_min_adv[inter_positions].mean().item()
                                                
                                            # Only Student Advantage
                                            global_mask_only_stu = (overlap_mask < 0.5) & global_valid_mask_bool_k
                                            if global_mask_only_stu.any():
                                                global_avg_adv_only_stu = advantages[global_mask_only_stu].mean()
                                                metrics["val-topk/adv_only_stu"] = global_avg_adv_only_stu.item()

                                            chunk_size = 1024
                                            
                                            # We can iterate up to max_len
                                            for start_idx in range(0, max_len, chunk_size):
                                                end_idx = min(start_idx + chunk_size, max_len)
                                                chunk_key = f"{start_idx}_{end_idx}"
                                                
                                                # Slice tensors
                                                chunk_response_mask = response_mask[:, start_idx:end_idx].bool() # (BS, Chunk)
                                                chunk_overlap_mask = overlap_mask[:, start_idx:end_idx] # (BS, Chunk, K)
                                                chunk_adv = advantages[:, start_idx:end_idx] # (BS, Chunk, K)
                                                
                                                if not chunk_response_mask.any():
                                                    continue
                                                
                                                # Expand response mask to K for element-wise ops
                                                chunk_valid_mask = chunk_response_mask.unsqueeze(-1).expand_as(chunk_overlap_mask)
                                                
                                                # 1. Overlap Ratio per chunk
                                                total_valid_k = chunk_valid_mask.sum()
                                                total_overlap_k = (chunk_overlap_mask * chunk_valid_mask.float()).sum()
                                                
                                                if total_valid_k > 0:
                                                    metrics[f"val-topk/overlap_ratio_chunk_{chunk_key}"] = (total_overlap_k / total_valid_k).item()
                                                
                                                # 2. Advantage Analysis
                                                # Intersection Advantage
                                                mask_inter = (chunk_overlap_mask > 0.5) & chunk_valid_mask
                                                if mask_inter.any():
                                                    avg_adv_inter = chunk_adv[mask_inter].mean()
                                                    metrics[f"val-topk/adv_intersection_chunk_{chunk_key}"] = avg_adv_inter.item()
                                                    
                                                    # Compute p metrics for intersection chunk (only_stu/intersection strategy)
                                                    if student_log_probs is not None and teacher_on_stu_log_probs is not None:
                                                        chunk_student_lp = student_log_probs[:, start_idx:end_idx]
                                                        chunk_teacher_lp = teacher_on_stu_log_probs[:, start_idx:end_idx]
                                                        student_p_c = torch.exp(chunk_student_lp)
                                                        teacher_p_c = torch.exp(chunk_teacher_lp)
                                                        inter_pos_c = mask_inter.any(dim=-1)
                                                        
                                                        # p_sum metrics
                                                        student_p_masked_c = torch.where(mask_inter, student_p_c, torch.zeros_like(student_p_c))
                                                        teacher_p_masked_c = torch.where(mask_inter, teacher_p_c, torch.zeros_like(teacher_p_c))
                                                        metrics[f"val-topk/student_p_sum_intersection_chunk_{chunk_key}"] = student_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_sum_intersection_chunk_{chunk_key}"] = teacher_p_masked_c.sum(dim=-1)[inter_pos_c].mean().item()
                                                        
                                                        # max_p metrics
                                                        student_p_for_max_c = torch.where(mask_inter, student_p_c, torch.full_like(student_p_c, float('-inf')))
                                                        teacher_p_for_max_c = torch.where(mask_inter, teacher_p_c, torch.full_like(teacher_p_c, float('-inf')))
                                                        max_stu_idx_c = student_p_for_max_c.argmax(dim=-1)
                                                        max_tch_idx_c = teacher_p_for_max_c.argmax(dim=-1)
                                                        
                                                        max_stu_p_c = student_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_stu_c = teacher_p_c.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_stu_c = chunk_adv.gather(-1, max_stu_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        max_tch_p_c = teacher_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_tch_c = student_p_c.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        adv_at_max_tch_c = chunk_adv.gather(-1, max_tch_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-topk/max_student_p_intersection_chunk_{chunk_key}"] = max_stu_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/teacher_p_at_max_student_intersection_chunk_{chunk_key}"] = tch_p_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_student_intersection_chunk_{chunk_key}"] = adv_at_max_stu_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/max_teacher_p_intersection_chunk_{chunk_key}"] = max_tch_p_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/student_p_at_max_teacher_intersection_chunk_{chunk_key}"] = stu_p_at_max_tch_c[inter_pos_c].mean().item()
                                                        metrics[f"val-topk/adv_at_max_teacher_intersection_chunk_{chunk_key}"] = adv_at_max_tch_c[inter_pos_c].mean().item()
                                                        
                                                        # max/min adv metrics
                                                        adv_for_max_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('-inf')))
                                                        adv_for_min_c = torch.where(mask_inter, chunk_adv, torch.full_like(chunk_adv, float('inf')))
                                                        max_adv_idx_c = adv_for_max_c.argmax(dim=-1)
                                                        min_adv_idx_c = adv_for_min_c.argmin(dim=-1)
                                                        
                                                        max_adv_c = chunk_adv.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_max_adv_c = student_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_max_adv_c = teacher_p_c.gather(-1, max_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        min_adv_c = chunk_adv.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        stu_p_at_min_adv_c = student_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        tch_p_at_min_adv_c = teacher_p_c.gather(-1, min_adv_idx_c.unsqueeze(-1)).squeeze(-1)
                                                        
                                                        metrics[f"val-extrema/max_adv_intersection_chunk_{chunk_key}"] = max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_max_adv_intersection_chunk_{chunk_key}"] = stu_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_max_adv_intersection_chunk_{chunk_key}"] = tch_p_at_max_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/min_adv_intersection_chunk_{chunk_key}"] = min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/student_p_at_min_adv_intersection_chunk_{chunk_key}"] = stu_p_at_min_adv_c[inter_pos_c].mean().item()
                                                        metrics[f"val-extrema/teacher_p_at_min_adv_intersection_chunk_{chunk_key}"] = tch_p_at_min_adv_c[inter_pos_c].mean().item()
                                                    
                                                # Only Student Advantage
                                                mask_only_stu = (chunk_overlap_mask < 0.5) & chunk_valid_mask
                                                if mask_only_stu.any():
                                                    avg_adv_only_stu = chunk_adv[mask_only_stu].mean()
                                                    metrics[f"val-topk/adv_only_stu_chunk_{chunk_key}"] = avg_adv_only_stu.item()
                                            
                            except Exception as e:
                                print(f"Error computing Top-K metrics: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    if self.config.trainer.get("is_plot", False) and (self.global_steps == 1 or self.global_steps % 10 == 0):
                        try:
                            import matplotlib.pyplot as plt
                            try:
                                import swanlab as _swanlab2
                            except ImportError:
                                _swanlab2 = None

                            # Check if teacher_entropy is available
                            if "teacher_entropy" in batch.batch.keys():
                                teacher_entropy = batch.batch["teacher_entropy"]
                                
                                # Determine advantage to use
                                if "token_level_advantage_direct" in batch.batch.keys():
                                    adv = batch.batch["token_level_advantage_direct"]
                                else:
                                    adv = batch.batch["advantages"]

                                if adv.dim() == 3:
                                    adv = adv.sum(dim=-1)
                                
                                response_mask = batch.batch["response_mask"]
                                
                                # Move to CPU and detach
                                teacher_entropy_cpu = teacher_entropy.detach().cpu()
                                adv_cpu = adv.detach().cpu()
                                mask_cpu = response_mask.detach().cpu().bool()
                                
                                # Create position indices
                                batch_size, seq_len = teacher_entropy_cpu.shape
                                positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
                                
                                # Filter using mask
                                valid_indices = mask_cpu
                                valid_positions = positions[valid_indices].numpy()
                                valid_entropy = teacher_entropy_cpu[valid_indices].numpy()
                                valid_adv = adv_cpu[valid_indices].numpy()
                                
                                # 1. Plot Teacher Entropy Scatter
                                plt.figure(figsize=(10, 6))
                                plt.scatter(valid_positions, valid_entropy, alpha=0.05, s=1)
                                plt.title(f"Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Teacher Entropy")
                                plt.tight_layout()
                                entropy_plot = _swanlab2.Image(plt, caption=f"Teacher Entropy vs Position (Step {self.global_steps})") if _swanlab2 is not None else None
                                plt.close()

                                # 2. Plot Advantage Scatter
                                plt.figure(figsize=(10, 6))
                                plt.scatter(valid_positions, valid_adv, alpha=0.05, s=1)
                                plt.title(f"Advantage vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Advantage")
                                plt.tight_layout()
                                adv_plot = _swanlab2.Image(plt, caption=f"Advantage vs Position (Step {self.global_steps})") if _swanlab2 is not None else None
                                plt.close()

                                # Compute Average per Position
                                # Need to handle masking correctly.
                                # Use float tensor for mask to sum counts
                                mask_float = mask_cpu.float()
                                
                                # Sum values per position
                                sum_entropy = (teacher_entropy_cpu * mask_float).sum(dim=0)
                                sum_adv = (adv_cpu * mask_float).sum(dim=0)
                                count_per_pos = mask_float.sum(dim=0)
                                
                                # Avoid division by zero
                                valid_pos_mask = count_per_pos > 0
                                avg_entropy = torch.zeros_like(sum_entropy)
                                avg_adv = torch.zeros_like(sum_adv)
                                
                                avg_entropy[valid_pos_mask] = sum_entropy[valid_pos_mask] / count_per_pos[valid_pos_mask]
                                avg_adv[valid_pos_mask] = sum_adv[valid_pos_mask] / count_per_pos[valid_pos_mask]

                                # --- New Split Advantage Plots ---
                                avg_adv_inter = None
                                avg_adv_only_stu = None
                                overlap_mask_cpu = None
                                
                                if "overlap_mask" in batch.batch.keys():
                                    overlap_mask_cpu = batch.batch["overlap_mask"].detach().cpu()

                                if overlap_mask_cpu is not None and adv_cpu.dim() == 3:
                                    # Calculate Avg Advantage per Position for Intersection
                                    # overlap_mask_cpu: (BS, SeqLen, K)
                                    # adv_cpu: (BS, SeqLen, K)
                                    # mask_cpu: (BS, SeqLen)
                                    
                                    # Expand mask_cpu to K
                                    mask_cpu_k = mask_cpu.unsqueeze(-1).expand_as(overlap_mask_cpu)
                                    
                                    # Intersection
                                    mask_inter = (overlap_mask_cpu > 0.5) & mask_cpu_k
                                    
                                    # We sum over Batch AND K for each position
                                    sum_adv_inter = (adv_cpu * mask_inter.float()).sum(dim=(0, 2))
                                    count_inter = mask_inter.float().sum(dim=(0, 2))
                                    
                                    avg_adv_inter = torch.zeros(seq_len)
                                    valid_inter = count_inter > 0
                                    avg_adv_inter[valid_inter] = sum_adv_inter[valid_inter] / count_inter[valid_inter]
                                    
                                    # Only Stu
                                    mask_only_stu = (overlap_mask_cpu < 0.5) & mask_cpu_k
                                    
                                    sum_adv_only_stu = (adv_cpu * mask_only_stu.float()).sum(dim=(0, 2))
                                    count_only_stu = mask_only_stu.float().sum(dim=(0, 2))
                                    
                                    avg_adv_only_stu = torch.zeros(seq_len)
                                    valid_only_stu = count_only_stu > 0
                                    avg_adv_only_stu[valid_only_stu] = sum_adv_only_stu[valid_only_stu] / count_only_stu[valid_only_stu]
                                
                                # Convert to numpy for plotting
                                # We only plot positions that have at least one valid token
                                # Find the max position index that has valid data
                                if valid_pos_mask.any():
                                    max_valid_pos = torch.where(valid_pos_mask)[0].max().item()
                                    plot_positions = torch.arange(max_valid_pos + 1).numpy()
                                    plot_avg_entropy = avg_entropy[:max_valid_pos + 1].numpy()
                                    plot_avg_adv = avg_adv[:max_valid_pos + 1].numpy()
                                    
                                    plot_avg_adv_inter = avg_adv_inter[:max_valid_pos + 1].numpy() if avg_adv_inter is not None else None
                                    plot_avg_adv_only_stu = avg_adv_only_stu[:max_valid_pos + 1].numpy() if avg_adv_only_stu is not None else None
                                else:
                                    plot_positions = np.array([])
                                    plot_avg_entropy = np.array([])
                                    plot_avg_adv = np.array([])
                                    plot_avg_adv_inter = None
                                    plot_avg_adv_only_stu = None

                                # 3. Plot Average Teacher Entropy Line
                                plt.figure(figsize=(10, 6))
                                plt.plot(plot_positions, plot_avg_entropy)
                                plt.title(f"Avg Teacher Entropy vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Teacher Entropy")
                                plt.grid(True)
                                plt.tight_layout()
                                avg_entropy_plot = _swanlab2.Image(plt, caption=f"Avg Teacher Entropy vs Position (Step {self.global_steps})") if _swanlab2 is not None else None
                                plt.close()

                                # 4. Plot Average Advantage Line
                                plt.figure(figsize=(10, 6))
                                plt.plot(plot_positions, plot_avg_adv, label="Total")
                                if plot_avg_adv_inter is not None:
                                    plt.plot(plot_positions, plot_avg_adv_inter, label="Intersection")
                                if plot_avg_adv_only_stu is not None:
                                    plt.plot(plot_positions, plot_avg_adv_only_stu, label="Only Stu")

                                plt.title(f"Avg Advantage vs Position (Step {self.global_steps})")
                                plt.xlabel("Position")
                                plt.ylabel("Avg Advantage")
                                plt.legend()
                                plt.grid(True)
                                plt.tight_layout()
                                avg_adv_plot = _swanlab2.Image(plt, caption=f"Avg Advantage vs Position (Step {self.global_steps})") if _swanlab2 is not None else None
                                plt.close()

                                # Log to SwanLab (only if swanlab is available and all plots were created)
                                if _swanlab2 is not None and all(
                                    p is not None for p in [entropy_plot, adv_plot, avg_entropy_plot, avg_adv_plot]
                                ):
                                    _swanlab2.log({
                                        "viz/teacher_entropy_scatter": entropy_plot,
                                        "viz/advantage_scatter": adv_plot,
                                        "viz/avg_teacher_entropy_line": avg_entropy_plot,
                                        "viz/avg_advantage_line": avg_adv_plot
                                    }, step=self.global_steps)
                                    print(f"Logged 4 plots to SwanLab at step {self.global_steps}.")
                                else:
                                    _logger.debug(
                                        "[viz] swanlab not installed; skipping entropy/advantage plots at step %d.",
                                        self.global_steps,
                                    )

                                # Free memory
                                del teacher_entropy_cpu, adv_cpu, mask_cpu, mask_float
                                del valid_positions, valid_entropy, valid_adv, positions
                                del sum_entropy, sum_adv, count_per_pos, avg_entropy, avg_adv
                                del plot_positions, plot_avg_entropy, plot_avg_adv
                                del entropy_plot, adv_plot, avg_entropy_plot, avg_adv_plot
                            else:
                                print("teacher_entropy not found in batch. Skipping plot.")
                                
                        except Exception as e:
                            print(f"Error plotting/logging: {e}")
                            import traceback
                            traceback.print_exc()

                    # Pop unused keys to save memory before PPO update
                    keys_to_pop = [
                        "teacher_on_student_log_probs",
                        "teacher_top_k_ids",
                        "teacher_top_k_log_probs",
                        "teacher_entropy",
                        "overlap_mask",
                        "teacher_in_student_mask",
                        "student_log_probs_on_teacher_ids",
                    ]
                    for key in keys_to_pop:
                        if key in batch.batch.keys():
                            batch.batch.pop(key)

                    # ------------------------------------------------------------------ #
                    # Actor / Critic DP alignment                                         #
                    #                                                                     #
                    # After Teacher routing + strip_dummy_samples the batch may have a   #
                    # size that is not divisible by the Actor DP world-size (e.g. 58     #
                    # real samples with DP=8 → 58 % 8 = 2 ≠ 0).  Pad with dummy rows    #
                    # before dispatching to Actor/Critic workers, then unpad the result. #
                    # This mirrors the same pattern used for compute_distillation_reward. #
                    # ------------------------------------------------------------------ #
                    _actor_dp_size = int(self.actor_rollout_wg.world_size)
                    advance_lr_scheduler = (
                        not self._ff_opd_enabled
                        or (
                            ff_phase == "fresh"
                            and self._ff_fresh_step + 1 < self.fresh_total_steps
                        )
                    )
                    batch.meta_info["advance_lr_scheduler"] = advance_lr_scheduler
                    if self._ff_opd_enabled:
                        metrics["ff/lr_scheduler_advanced"] = float(advance_lr_scheduler)
                    batch, _actor_pad_size = pad_dataproto_to_divisor(batch, size_divisor=_actor_dp_size)
                    _ff_ht_enabled = bool(batch.meta_info.get("ff_opd_ht_enabled", False))
                    if (self._tlr_opd_enabled or _ff_ht_enabled) and _actor_pad_size > 0:
                        # pad_dataproto_to_divisor repeats real rows.  Convert
                        # those repeats into true zero-loss Actor dummies so
                        # only the selected BoN-4 trajectories contribute.
                        for key in (
                            "response_mask",
                            "advantages",
                            "returns",
                            "token_level_scores",
                            "token_level_rewards",
                        ):
                            if key in batch.batch:
                                batch.batch[key][-_actor_pad_size:] = 0
                    if _ff_ht_enabled:
                        batch.meta_info["ff_opd_ht_actor_batch_count"] = len(batch)
                    if _actor_pad_size > 0:
                        _logger.info(
                            "[Actor DP alignment] batch_before=%d  dummy_added=%d  "
                            "batch_after=%d  actor_dp=%d",
                            len(batch) - _actor_pad_size,
                            _actor_pad_size,
                            len(batch),
                            _actor_dp_size,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if self._ff_opd_enabled:
                            self._ff_optimizer_step += 1
                            self._ff_opd_manager.student_model_version = str(self._ff_optimizer_step)
                            metrics["ff/actual_optimizer_steps"] = 1.0
                            metrics["ff/optimizer_step"] = float(self._ff_optimizer_step)
                            metrics["train/optimizer_update_count"] = 1.0
                            # The FF objective is the reverse-KL OPD policy loss.
                            opd_loss = actor_output_metrics.get("actor/pg_loss", "")
                            for record in ff_opd_context.records:
                                if record["opd_loss_contributed"]:
                                    record["optimizer_step"] = self._ff_optimizer_step
                                    record["optimizer_updated"] = 1
                                    record["opd_loss"] = opd_loss
                            if self._ff_opd_manager.config.debug_assertions:
                                assert all(
                                    record["optimizer_updated"] == 0
                                    for record in ff_opd_context.records
                                    if record["teacher_queried"] == 0
                                )
                            self._ff_opd_csv.append(ff_opd_context.records)

                    # Strip Actor DP padding so downstream code (metrics, logging,
                    # rollout data) sees only the real samples.
                    if _actor_pad_size > 0:
                        batch = unpad_dataproto(batch, pad_size=_actor_pad_size)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                save_only_final = self.config.trainer.get("save_only_final", False)
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or (
                        not save_only_final
                        and (self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration)
                    )
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                if self._ff_opd_enabled:
                    teacher_gpu_count = int(getattr(self.rm_wg, "world_size", n_gpus)) if self.use_rm else 0
                    metrics["teacher/gpu_hours"] = (
                        teacher_gpu_count * float(timing_raw.get("compute_rm_score", 0.0)) / 3600.0
                    )
                    metrics["train/e2e_gpu_hours"] = n_gpus * float(timing_raw.get("step", 0.0)) / 3600.0
                if self._tlr_opd_enabled:
                    metrics["tlr/e2e_gpu_hours"] = (
                        n_gpus * float(timing_raw.get("step", 0.0)) / 3600.0
                    )
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                self._write_opd_step_metrics(metrics, n_gpus)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                advance_batch_counters_and_progress()
                if (
                    self._ff_opd_enabled
                    and self._ff_opd_manager.config.boundary_opd is not None
                    and self._ff_opd_manager.config.fresh_step_limit > 0
                    and ff_phase == "fresh"
                ):
                    _logger.info(
                        "[PDA smoke] training step completed: %d/%d global_step=%d optimizer_steps=%d",
                        self._ff_fresh_step,
                        self.fresh_total_steps,
                        self.global_steps,
                        self._ff_optimizer_step,
                    )
                # Persist the one-pass model immediately. Retry ablations keep
                # this post-fresh checkpoint independently of their final one.
                if (
                    self._ff_opd_enabled
                    and ff_phase == "fresh"
                    and self._ff_fresh_step >= self.fresh_total_steps
                    and self.config.trainer.save_freq > 0
                ):
                    if self._ff_opd_manager.config.max_no_success_retries == 0:
                        _logger.info("[FF training complete] step=%d saving final checkpoint", self.global_steps)
                    else:
                        _logger.info(
                            "[FF fresh complete] step=%d saving checkpoint before retry rounds",
                            self.global_steps,
                        )
                    with marked_timer("save_fresh_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                    bucket_counts: dict[str, int] = defaultdict(int)
                    for state in self._ff_opd_manager.prompt_states.values():
                        bucket_counts[state.current_bucket] += 1
                    total_prompts = len(self._ff_opd_manager.prompt_states)
                    _logger.info(
                        "[FF fresh buckets] all_correct=%d frontier=%d no_success=%d "
                        "total_prompts=%d",
                        bucket_counts.get("all_correct", 0),
                        bucket_counts.get("frontier", 0),
                        bucket_counts.get("no_success", 0),
                        total_prompts,
                    )
                # Save one checkpoint at the exact end of each dataset epoch,
                # independent of trainer.save_freq's step-count modulo (which
                # can't be pre-aligned to epoch boundaries: the dataloader
                # length depends on prompt-length filtering, known only at
                # runtime). Opt-in via trainer.save_every_epoch so existing
                # step-frequency-based runs are unaffected.
                if (
                    not self._ff_opd_enabled
                    and self.config.trainer.get("save_every_epoch", False)
                    and self._epoch_step_idx >= general_steps_per_epoch
                    and self.config.trainer.save_freq > 0
                ):
                    _logger.info(
                        "[epoch checkpoint] epoch=%d/%d step=%d saving checkpoint",
                        epoch + 1,
                        self.config.trainer.total_epochs,
                        self.global_steps,
                    )
                    with marked_timer("save_epoch_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        if self._ff_opd_enabled:
            self._ff_opd_csv.flush()
            self._ff_opd_profiles.flush()
            if self._ff_kl_profiles is not None:
                self._ff_kl_profiles.flush()
            if (
                self.config.trainer.save_freq > 0
                and self._ff_opd_manager.config.max_no_success_retries > 0
            ):
                self._save_checkpoint()
            self._ff_opd_csv.close()
            self._ff_opd_profiles.close()
            if self._ff_kl_profiles is not None:
                self._ff_kl_profiles.close()
            progress_bar.close()
