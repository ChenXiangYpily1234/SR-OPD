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

"""Frozen transition or direct-state calibration artifacts for Boundary selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from verl.utils.boundary_opd import (
    BoundaryOPDSettings,
    stable_group_sibling_indices,
)

ARTIFACT_FORMAT = "boundary-transition-calibration-v1"
DIRECT_STATE_ARTIFACT_FORMAT = "boundary-direct-hidden-state-calibration-v1"
# The only score mode whose calibration lives on direct hidden states (one
# state per response boundary) instead of on hidden transitions.
DIRECT_STATE_SCORE_MODE = "persistent_departure_area"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or a directory tree including stable relative filenames."""

    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise FileNotFoundError(target)
    digest = hashlib.sha256()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot hash empty directory: {target}")
    for item in files:
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.suffix == ".json" else target / "manifest.json"


def _artifact_output_paths(path: str | Path) -> tuple[Path, Path, Path]:
    manifest_path = _manifest_path(path)
    return (
        manifest_path,
        manifest_path.with_name(f"{manifest_path.stem}.statistics.npz"),
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
    )


def _refuse_existing_artifact(path: str | Path) -> tuple[Path, Path, Path]:
    outputs = _artifact_output_paths(path)
    existing = [output for output in outputs if output.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite Boundary calibration artifact files: "
            + ", ".join(str(output) for output in existing)
        )
    return outputs


@dataclass(frozen=True)
class BoundaryCalibrationArtifact:
    mean: torch.Tensor
    cholesky: torch.Tensor
    manifest: dict[str, Any]
    manifest_sha256: str


def load_boundary_calibration(
    settings: BoundaryOPDSettings,
    *,
    expected_hidden_dim: int | None = None,
) -> BoundaryCalibrationArtifact:
    """Load and fully authenticate a frozen calibration artifact."""

    manifest_path = _manifest_path(settings.calibration_artifact_path)
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != settings.calibration_artifact_sha256.lower():
        raise ValueError(
            "Boundary calibration manifest SHA-256 mismatch: "
            f"expected={settings.calibration_artifact_sha256} actual={actual_manifest_sha}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_format = (
        DIRECT_STATE_ARTIFACT_FORMAT if settings.score_mode == DIRECT_STATE_SCORE_MODE else ARTIFACT_FORMAT
    )
    if manifest.get("format") != expected_format:
        raise ValueError(f"unsupported Boundary calibration format: {manifest.get('format')!r}")
    expected_domain = (
        "direct_hidden_state" if settings.score_mode == DIRECT_STATE_SCORE_MODE else "hidden_transition"
    )
    if manifest.get("representation_domain", "hidden_transition") != expected_domain:
        raise ValueError(
            f"Boundary calibration representation_domain mismatch: expected={expected_domain!r} "
            f"actual={manifest.get('representation_domain')!r}"
        )
    for key, expected in (
        ("num_boundaries", settings.num_boundaries),
        ("prompt_count", settings.calibration_num_prompts),
        ("seed", 42),
        ("whitening_regularization", settings.whitening_regularization),
    ):
        actual = manifest.get(key)
        if actual != expected:
            raise ValueError(f"Boundary calibration {key} mismatch: expected={expected!r} actual={actual!r}")
    if settings.calibration_model_hash and manifest.get("model_hash") != settings.calibration_model_hash:
        raise ValueError("Boundary calibration model hash does not match the current Student checkpoint")
    if settings.calibration_data_hash and manifest.get("data_hash") != settings.calibration_data_hash:
        raise ValueError("Boundary calibration data hash does not match the current training dataset")
    hidden_dim = int(manifest.get("hidden_dim", 0))
    prompt_ids = manifest.get("prompt_ids")
    rollout_sample_ids = manifest.get("rollout_sample_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) != int(manifest["prompt_count"]):
        raise ValueError("Boundary calibration prompt IDs do not match prompt_count")
    if not isinstance(rollout_sample_ids, list) or len(rollout_sample_ids) != int(manifest.get("rollout_count", -1)):
        raise ValueError("Boundary calibration rollout sample IDs do not match rollout_count")
    subset_hash = hashlib.sha256("\n".join(map(str, rollout_sample_ids)).encode("utf-8")).hexdigest()
    if subset_hash != manifest.get("subset_ids_sha256"):
        raise ValueError("Boundary calibration subset ID hash mismatch")
    if expected_hidden_dim is not None and hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"Boundary calibration hidden_dim mismatch: expected={expected_hidden_dim} actual={hidden_dim}"
        )
    statistics_path = manifest_path.parent / str(manifest.get("statistics_file", ""))
    if sha256_file(statistics_path) != manifest.get("statistics_sha256"):
        raise ValueError("Boundary calibration statistics file SHA-256 mismatch")
    with np.load(statistics_path, allow_pickle=False) as payload:
        mean_array = np.asarray(payload["mean"], dtype=np.float32)
        cholesky_array = np.asarray(payload["cholesky"], dtype=np.float32)
    if mean_array.shape != (hidden_dim,) or cholesky_array.shape != (hidden_dim, hidden_dim):
        raise ValueError("Boundary calibration tensor shapes do not match hidden_dim")
    if _tensor_sha256(mean_array) != manifest.get("mean_sha256"):
        raise ValueError("Boundary calibration mean tensor hash mismatch")
    if _tensor_sha256(cholesky_array) != manifest.get("cholesky_sha256"):
        raise ValueError("Boundary calibration Cholesky tensor hash mismatch")
    mean = torch.from_numpy(mean_array.copy()).detach()
    cholesky = torch.from_numpy(cholesky_array.copy()).detach()
    if not torch.isfinite(mean).all() or not torch.isfinite(cholesky).all():
        raise FloatingPointError("Boundary calibration contains NaN or Inf")
    return BoundaryCalibrationArtifact(mean, cholesky, manifest, actual_manifest_sha)


class BoundaryCalibrationAccumulator:
    """FP64 streaming mean/covariance collector over the configured representation domain."""

    def __init__(self, settings: BoundaryOPDSettings, *, k_rollouts: int, seed: int = 42):
        if not settings.calibration_collect:
            raise ValueError("BoundaryCalibrationAccumulator requires calibration_collect=true")
        self.settings = settings
        self.k_rollouts = int(k_rollouts)
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError("Boundary calibration seed must be non-negative")
        _refuse_existing_artifact(settings.calibration_artifact_path)
        self.prompt_ids: list[str] = []
        self.rollout_sample_ids: list[str] = []
        self.sample_count = 0
        self.sum: torch.Tensor | None = None
        self.cross: torch.Tensor | None = None
        self.hidden_dim: int | None = None
        self._seen_prompts: set[str] = set()
        self.artifact: BoundaryCalibrationArtifact | None = None

    @property
    def complete(self) -> bool:
        return len(self.prompt_ids) >= self.settings.calibration_num_prompts

    def update(
        self,
        boundary_states: torch.Tensor,
        transition_valid_mask: torch.Tensor,
        prompt_ids: Sequence[Any],
        rollout_ids: Sequence[Any],
    ) -> int:
        if boundary_states.ndim != 3:
            raise ValueError("calibration boundary_states must have shape [B, S, D]")
        expected = (boundary_states.shape[0], self.settings.num_boundaries)
        if transition_valid_mask.shape != expected:
            raise ValueError(f"calibration transition_valid_mask must have shape {expected}")
        groups = stable_group_sibling_indices(prompt_ids, rollout_ids, expected_siblings=self.k_rollouts)
        accepted_rows: list[int] = []
        for prompt_id, indices in groups:
            if self.complete:
                break
            if prompt_id in self._seen_prompts:
                raise ValueError(f"duplicate calibration prompt id: {prompt_id}")
            self._seen_prompts.add(prompt_id)
            self.prompt_ids.append(prompt_id)
            accepted_rows.extend(indices)
            self.rollout_sample_ids.extend(f"{prompt_id}:{int(rollout_ids[index])}" for index in indices)
        if not accepted_rows:
            return 0
        row_index = torch.as_tensor(accepted_rows, dtype=torch.long, device=boundary_states.device)
        states = boundary_states.index_select(0, row_index).detach().to(device="cpu", dtype=torch.float64)
        valid = transition_valid_mask.index_select(0, row_index).detach().to(device="cpu", dtype=torch.bool)
        if self.settings.score_mode == DIRECT_STATE_SCORE_MODE:
            if states.shape[1] != self.settings.num_boundaries:
                raise ValueError("PDA calibration requires M direct hidden states")
            samples = states[valid]
        else:
            if states.shape[1] != self.settings.num_boundaries + 1:
                raise ValueError("transition calibration requires M+1 hidden states")
            transitions = states[:, 1:, :] - states[:, :-1, :]
            samples = transitions[valid]
        if samples.numel() == 0:
            raise ValueError("calibration subset contains no valid representation samples")
        hidden_dim = int(samples.shape[-1])
        if self.hidden_dim is None:
            self.hidden_dim = hidden_dim
            self.sum = torch.zeros(hidden_dim, dtype=torch.float64)
            self.cross = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64)
        elif hidden_dim != self.hidden_dim:
            raise ValueError("calibration hidden dimension changed across batches")
        assert self.sum is not None and self.cross is not None
        self.sum.add_(samples.sum(dim=0))
        self.cross.add_(samples.T @ samples)
        self.sample_count += int(samples.shape[0])
        return len(accepted_rows)

    def finalize(self) -> BoundaryCalibrationArtifact:
        if not self.complete:
            raise RuntimeError(
                "Boundary calibration ended before the fixed prompt subset was complete: "
                f"collected={len(self.prompt_ids)} required={self.settings.calibration_num_prompts}"
            )
        if self.sample_count < 2 or self.sum is None or self.cross is None or self.hidden_dim is None:
            raise RuntimeError("Boundary calibration requires at least two valid representation samples")
        mean = self.sum / self.sample_count
        covariance = (self.cross - self.sample_count * torch.outer(mean, mean)) / (self.sample_count - 1)
        covariance = 0.5 * (covariance + covariance.T)
        regularized = covariance + self.settings.whitening_regularization * torch.eye(
            self.hidden_dim, dtype=torch.float64
        )
        cholesky = torch.linalg.cholesky(regularized)
        mean_array = mean.to(torch.float32).numpy()
        cholesky_array = cholesky.to(torch.float32).numpy()
        manifest_path, statistics_path, checksum_path = _refuse_existing_artifact(
            self.settings.calibration_artifact_path
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(statistics_path, mean=mean_array, cholesky=cholesky_array)
        subset_hash = hashlib.sha256(
            "\n".join(self.rollout_sample_ids).encode("utf-8")
        ).hexdigest()
        direct_states = self.settings.score_mode == DIRECT_STATE_SCORE_MODE
        manifest = {
            "format": DIRECT_STATE_ARTIFACT_FORMAT if direct_states else ARTIFACT_FORMAT,
            "representation_domain": "direct_hidden_state" if direct_states else "hidden_transition",
            "uses_hidden_difference": not direct_states,
            "model_hash": self.settings.calibration_model_hash,
            "data_hash": self.settings.calibration_data_hash,
            "seed": self.seed,
            "num_boundaries": self.settings.num_boundaries,
            "prompt_count": len(self.prompt_ids),
            "rollout_count": len(self.rollout_sample_ids),
            "sample_count": self.sample_count,
            "hidden_dim": self.hidden_dim,
            "whitening_regularization": self.settings.whitening_regularization,
            "subset_ids_sha256": subset_hash,
            "prompt_ids": self.prompt_ids,
            "rollout_sample_ids": self.rollout_sample_ids,
            "statistics_file": statistics_path.name,
            "statistics_sha256": sha256_file(statistics_path),
            "mean_sha256": _tensor_sha256(mean_array),
            "cholesky_sha256": _tensor_sha256(cholesky_array),
            "accumulation_dtype": "float64",
            "covariance_estimator": "unbiased",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_sha = sha256_file(manifest_path)
        checksum_path.write_text(
            f"{manifest_sha}  {manifest_path.name}\n", encoding="utf-8"
        )
        load_settings = BoundaryOPDSettings(
            **{
                **self.settings.__dict__,
                "calibration_collect": False,
                "similarity_metric": (
                    "centered_hidden_state_cosine" if direct_states else "centered_hidden_cosine"
                ),
                "calibration_artifact_path": str(manifest_path),
                "calibration_artifact_sha256": manifest_sha,
            }
        )
        self.artifact = load_boundary_calibration(load_settings, expected_hidden_dim=self.hidden_dim)
        return self.artifact


def _validate_cli(args: argparse.Namespace) -> None:
    model_hash = sha256_path(args.model_path) if args.model_path else args.model_hash
    data_hash = sha256_path(args.data_path) if args.data_path else args.data_hash
    direct_states = args.representation_domain == "direct_hidden_state"
    settings = BoundaryOPDSettings(
        similarity_metric=(
            "centered_hidden_state_cosine" if direct_states else "centered_hidden_cosine"
        ),
        calibration_artifact_path=args.manifest,
        calibration_artifact_sha256=args.manifest_sha256,
        calibration_model_hash=model_hash or "",
        calibration_data_hash=data_hash or "",
        num_boundaries=args.num_boundaries,
        whitening_regularization=args.whitening_regularization,
        score_mode="persistent_departure_area" if direct_states else "legacy",
    )
    artifact = load_boundary_calibration(settings, expected_hidden_dim=args.hidden_dim)
    print(
        json.dumps(
            {
                "manifest": str(_manifest_path(args.manifest)),
                "manifest_sha256": artifact.manifest_sha256,
                "model_hash": artifact.manifest["model_hash"],
                "data_hash": artifact.manifest["data_hash"],
                "sample_count": artifact.manifest["sample_count"],
                "hidden_dim": artifact.manifest["hidden_dim"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hash_parser = subparsers.add_parser("hash", help="hash a model directory or dataset file")
    hash_parser.add_argument("path")
    validate = subparsers.add_parser("validate", help="authenticate a calibration artifact")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--manifest-sha256", required=True)
    validate.add_argument("--model-path")
    validate.add_argument("--data-path")
    validate.add_argument("--model-hash")
    validate.add_argument("--data-hash")
    validate.add_argument("--num-boundaries", type=int, default=16)
    validate.add_argument("--hidden-dim", type=int)
    validate.add_argument("--whitening-regularization", type=float, default=1.0e-4)
    validate.add_argument("--require-cholesky", action="store_true")
    validate.add_argument(
        "--representation-domain",
        choices=("hidden_transition", "direct_hidden_state"),
        default="hidden_transition",
    )
    args = parser.parse_args()
    if args.command == "hash":
        print(sha256_path(args.path))
    else:
        _validate_cli(args)


if __name__ == "__main__":
    main()
