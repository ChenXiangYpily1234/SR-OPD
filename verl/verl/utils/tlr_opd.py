"""Pre-Teacher trajectory routing for Selective OPSD (TLR paper form).

For every prompt, TLR selects exactly one valid Student rollout before the
Teacher forward, without inspecting verifier correctness.

Paper score (teacher-free disagreement proxy):

    H_i = mean_t H(Student(. | x, y_i,<t))     # mean Student token entropy
    L_i = |y_i|                                 # rollout length
    Hhat_i, Lhat_i = per-group min-max normalization into [0, 1]
    S_i = (1 - Lhat_i) * (1 - Hhat_i)           # prefer SHORT + LOW entropy

and the rollout with the highest S_i is sent to the privileged Teacher:
short and low-entropy rollouts win; long or high-entropy rollouts are
rejected. The original per-trajectory OPSD objective is retained; there is
no learned router, stochastic exploration, IPW, or Horvitz-Thompson
correction.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

TLR_MODE = "short_low_entropy"

# Minimum rollout length (valid tokens) for a trajectory to enter ScoreLH
# routing. Immediate-EOS degenerates (1-3 tokens) are simultaneously the
# shortest and the lowest-entropy sibling, so without this floor ScoreLH hits
# its theoretical maximum S=1 on them every time and the Student collapses
# into "stop immediately" within tens of steps (observed: candidate pool
# 0% -> ~100% length<16 over ~50 steps, matching the historical bon4_lh
# implementation note). This names the constraint precisely: ScoreLH is only
# defined/applied on candidates with minimal trajectory support; it does NOT
# declare shorter generations invalid data. Set to 0 to disable the floor
# (failure-mode ablation only).
TLR_MIN_ROLLOUT_TOKENS = 16

TLR_CSV_FIELDS = [
    "run_name",
    "seed",
    "global_step",
    "optimizer_step",
    "prompt_uid",
    "source_index",
    "rollout_idx",
    "eligible",
    "selected",
    "selection_mode",
    "response_length",
    "response_valid_tokens",
    "student_entropy_mean",
    "score_tlr",
    "teacher_queried",
    "teacher_input_tokens",
    "teacher_scored_tokens",
    "sampled_logratio_mean",
    "abs_sampled_logratio_mean",
    "sampled_reverse_kl_sum",
    "sampled_reverse_kl_token_mean",
    "actor_loss_contributed",
]


@dataclass
class TLRConfig:
    enabled: bool = False
    rollouts_per_prompt: int = 4
    save_csv: bool = True
    csv_path: str = ""
    seed: int = 42
    min_rollout_tokens: int = TLR_MIN_ROLLOUT_TOKENS

    @classmethod
    def from_mapping(cls, values: Mapping) -> TLRConfig:
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in dict(values).items() if key in known})

    def validate(self, rollout_n: int | None = None) -> None:
        if self.rollouts_per_prompt < 2:
            raise ValueError("Selective OPSD requires tlr_opd.rollouts_per_prompt>=2 (BoN-K)")
        if self.min_rollout_tokens < 0:
            raise ValueError("tlr_opd.min_rollout_tokens must be >=0 (0 disables the floor)")
        if rollout_n is not None and rollout_n != self.rollouts_per_prompt:
            raise ValueError(
                f"Selective OPSD requires actor_rollout_ref.rollout.n=tlr_opd.rollouts_per_prompt "
                f"({self.rollouts_per_prompt})"
            )


@dataclass
class TLRSelection:
    selected_indices: torch.Tensor
    scores: torch.Tensor
    eligible_mask: torch.Tensor
    prompt_inverse: torch.Tensor
    prompt_uids: list[str]


def _stable_prompt_groups(prompt_ids: Sequence[object]) -> tuple[list[str], torch.Tensor]:
    uid_to_group: dict[str, int] = {}
    ordered: list[str] = []
    inverse = []
    for raw_uid in prompt_ids:
        uid = str(raw_uid)
        if uid not in uid_to_group:
            uid_to_group[uid] = len(ordered)
            ordered.append(uid)
        inverse.append(uid_to_group[uid])
    return ordered, torch.tensor(inverse, dtype=torch.long)


def select_tlr_trajectories(
    *,
    prompt_ids: Sequence[object],
    eligible_mask: torch.Tensor,
    response_lengths: torch.Tensor,
    student_entropy_mean: torch.Tensor,
    rollouts_per_prompt: int = 4,
    min_rollout_tokens: int = TLR_MIN_ROLLOUT_TOKENS,
) -> TLRSelection:
    """Select one eligible rollout per prompt by the paper TLR score.

    Per prompt group the length and mean entropy are min-max normalized into
    [0, 1]; the rollout maximizing

        S_i = (1 - Lhat_i) * (1 - Hhat_i)

    (shortest + lowest entropy) wins. Ties follow ``torch.argmax`` and
    therefore resolve to the earliest rollout index inside the prompt.  This
    makes routing deterministic and independent of distributed worker RNG
    state.

    ``min_rollout_tokens`` (>=1) is applied BEFORE normalization: trajectories
    shorter than the floor are removed from the group, so they cannot corrupt
    the min/max normalizers, and only then is ScoreLH computed on the
    remaining candidates. A prompt whose group has no trajectory reaching the
    floor is skipped entirely (no Teacher query for it); we never fall back to
    ScoreLH over the degenerate set, because that would re-route gradients
    onto immediate-EOS exactly when the floor matters most.
    Set ``min_rollout_tokens=0`` to disable the floor (failure-mode ablation).
    """
    if min_rollout_tokens < 0:
        raise ValueError("min_rollout_tokens must be >=0 (0 disables the floor)")
    eligible = eligible_mask.detach().bool().cpu()
    lengths = response_lengths.detach().float().cpu()
    entropies = student_entropy_mean.detach().float().cpu()
    prompt_uids, inverse = _stable_prompt_groups(prompt_ids)
    n = eligible.numel()
    if any(tensor.numel() != n for tensor in (lengths, entropies, inverse)):
        raise ValueError(
            "prompt_ids, eligible_mask, response_lengths, and student_entropy_mean "
            "must have equal length"
        )
    group_counts = torch.bincount(inverse, minlength=len(prompt_uids))
    if group_counts.numel() and not torch.all(group_counts == rollouts_per_prompt):
        raise ValueError(
            f"Selective OPSD BoN-{rollouts_per_prompt} requires exactly "
            f"{rollouts_per_prompt} generated rollouts per prompt"
        )
    if torch.any(eligible & (~torch.isfinite(lengths) | (lengths <= 0))):
        raise ValueError("eligible rollout lengths must be finite and positive")
    if torch.any(eligible & ~torch.isfinite(entropies)):
        raise ValueError("eligible rollout entropies must be finite")

    nondegenerate = eligible & (lengths >= float(min_rollout_tokens))

    scores = torch.full((n,), float("-inf"), dtype=torch.float32)
    selected: list[int] = []

    for group in range(len(prompt_uids)):
        group_mask = inverse == group
        indices = torch.where(group_mask & eligible)[0]
        if indices.numel() == 0:
            continue
        candidates = torch.where(group_mask & nondegenerate)[0]
        if candidates.numel() == 0:
            # All rollouts of this prompt are below the trajectory floor:
            # ScoreLH is out of scope for this group. Skip the Teacher query;
            # never fall back to routing over immediate-EOS degenerates.
            continue
        group_lengths = lengths[candidates]
        group_entropies = entropies[candidates]
        # Per-group min-max normalization. A degenerate span (all candidates
        # share the same length or entropy) collapses that factor to a
        # constant, and selection falls back to the other factor.
        length_span = group_lengths.max() - group_lengths.min()
        entropy_span = group_entropies.max() - group_entropies.min()
        lhat = (
            (group_lengths - group_lengths.min()) / length_span
            if length_span > 0
            else torch.full_like(group_lengths, 0.5)
        )
        hhat = (
            (group_entropies - group_entropies.min()) / entropy_span
            if entropy_span > 0
            else torch.full_like(group_entropies, 0.5)
        )
        group_scores = (1.0 - lhat) * (1.0 - hhat)
        scores[candidates] = group_scores
        selected.append(int(candidates[torch.argmax(group_scores)].item()))

    return TLRSelection(
        selected_indices=torch.tensor(selected, dtype=torch.long),
        scores=scores,
        eligible_mask=nondegenerate,
        prompt_inverse=inverse,
        prompt_uids=prompt_uids,
    )


class TLRCSVWriter:
    def __init__(self, path: str):
        self.path = path

    def append(self, rows: list[dict]) -> None:
        if not self.path or not rows:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=TLR_CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


def validate_method_exclusivity(tlr_enabled: bool, **methods: bool) -> None:
    conflicts = [name for name, enabled in methods.items() if enabled]
    if tlr_enabled and conflicts:
        raise ValueError(
            "TLR-OPD is mutually exclusive with "
            + ", ".join(conflicts)
            + "; enable exactly one pre-Teacher query method"
        )
