# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Boundary-OPD configuration and launcher wiring tests (spec sections 3 and 21)."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.utils.boundary_opd_fixtures import REPO_ROOT
from verl.trainer.config.algorithm import AlgoConfig, BoundaryOPDConfig
from verl.utils.boundary_opd import BOUNDARY_SELECTOR_MODES, BoundaryOPDSettings
from verl.utils.frontier_selector import FF_SELECTOR_MODES

CONFIG_DIR = REPO_ROOT / "verl" / "verl" / "trainer" / "config"
CONFIG_FILES = (
    "ppo_trainer.yaml",
    "_generated_ppo_trainer.yaml",
    "_generated_ppo_megatron_trainer.yaml",
)
BOUNDARY_SCRIPTS = (
    "strategy/run_global_random.sh",
    "strategy/run_random_wrong.sh",
    "strategy/run_random_correct.sh",
    "strategy/run_shortest_wrong.sh",
    "main/run_sr_opd_m8.sh",
    "main/run_sr_opd_m16.sh",
    "main/run_sr_opd_m32.sh",
    "main/run_frontier_tlr.sh",
    "representation/run_calibration.sh",
)


def _algorithm_config(name: str):
    return OmegaConf.load(CONFIG_DIR / name).algorithm


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_yaml_boundary_block_matches_the_dataclasses(name):
    algorithm = _algorithm_config(name)
    yaml_keys = set(algorithm.boundary_opd.keys())

    assert yaml_keys == {field.name for field in fields(BoundaryOPDConfig)}
    assert yaml_keys - {"_target_"} == {field.name for field in fields(BoundaryOPDSettings)}
    assert algorithm.boundary_opd._target_ == "verl.trainer.config.BoundaryOPDConfig"
    assert algorithm.ff_selector_mode in FF_SELECTOR_MODES

    defaults = BoundaryOPDConfig()
    for field in fields(BoundaryOPDConfig):
        if field.name == "_target_":
            continue
        assert algorithm.boundary_opd[field.name] == getattr(defaults, field.name), field.name


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_yaml_boundary_block_builds_valid_runtime_settings(name):
    algorithm = _algorithm_config(name)

    settings = BoundaryOPDSettings.from_mapping(OmegaConf.to_container(algorithm.boundary_opd, resolve=True))
    settings.validate()

    assert settings.num_boundaries == 16
    assert settings.capture_location == "pre_lm_head"
    assert settings.hidden_layer == -1
    assert settings.detach_hidden is True
    assert settings.similarity_metric == "raw_hidden_cosine"
    assert settings.fallback_to_ff_cost is False
    assert settings.debug_store_pairwise is False


def test_algo_config_defaults_expose_the_boundary_block():
    config = AlgoConfig()

    assert config.ff_selector_mode == "boundary_opd"
    assert isinstance(config.boundary_opd, BoundaryOPDConfig)
    assert config.boundary_opd.num_boundaries == 16


def test_train_sh_overrides_every_boundary_key():
    train_sh = (REPO_ROOT / "bash" / "train" / "sr-opd.sh").read_text(encoding="utf-8")

    overridden = set(re.findall(r"algorithm\.boundary_opd\.([a-z0-9_]+)=", train_sh))
    assert overridden == {field.name for field in fields(BoundaryOPDSettings)}
    assert "algorithm.ff_selector_mode=${FF_SELECTOR_MODE:-boundary_opd}" in train_sh


def test_ff_sh_accepts_every_selector_mode():
    ff_sh = (REPO_ROOT / "bash" / "train" / "sr-opd.sh").read_text(encoding="utf-8")

    allowed = set()
    for line in ff_sh.splitlines():
        stripped = line.strip()
        if stripped.endswith(") ;;") and "|" in stripped:
            allowed.update(stripped.split(")")[0].split("|"))
    assert set(FF_SELECTOR_MODES) <= allowed
    for mode in BOUNDARY_SELECTOR_MODES:
        assert mode in allowed


def test_boundary_launcher_scripts_only_override_the_allowed_knobs():
    script_dir = REPO_ROOT / "bash" / "sr_opd"

    assert not (script_dir / "legacy").exists()

    for name in BOUNDARY_SCRIPTS:
        path = script_dir / name
        assert path.exists(), name
        assert path.stat().st_mode & 0o111, f"{name} must be executable"

    wrapper = (script_dir / "common" / "_run_sr_opd.sh").read_text(encoding="utf-8")
    exported = set(re.findall(r"export (BOUNDARY_OPD_[A-Z_]+)=", wrapper))
    assert exported == {
        "BOUNDARY_OPD_CALIBRATION_DATA_HASH",
        "BOUNDARY_OPD_CALIBRATION_MODEL_HASH",
        "BOUNDARY_OPD_NUM_BOUNDARIES",
        "BOUNDARY_OPD_FALLBACK_TO_FF_COST",
        "BOUNDARY_OPD_SIMILARITY_METRIC",
    }
    assert "FF_MAX_NO_SUCCESS_RETRIES" not in wrapper
    assert "bash/train/sr-opd.sh" in wrapper or "sr-opd.sh" in wrapper
    # No training hyper-parameter may be duplicated into the wrappers.
    for name in BOUNDARY_SCRIPTS:
        body = (script_dir / name).read_text(encoding="utf-8")
        assert "actor_rollout_ref" not in body
        assert "trainer.total_epochs" not in body

    base_config = (script_dir / "common" / "base_config.sh").read_text(encoding="utf-8")
    assert "boundary_ensure_calibration" in base_config
    assert "Qwen3-1.7B" not in base_config

    calibration = (script_dir / "representation" / "run_calibration.sh").read_text(encoding="utf-8")
    assert '4|8|16|32)' in calibration
    assert 'BOUNDARY_OPD_CALIBRATION_SEED:-42' in calibration
    assert 'sr-opd-calibration-m${NUM_BOUNDARIES}' in calibration
    assert "BOUNDARY_OPD_CALIBRATION_REPRESENTATION_DOMAIN" in calibration
    assert "export BOUNDARY_SELECTOR_SCORE_MODE=persistent_departure_area" in calibration
    assert not any(
        line.startswith("exec bash") for line in calibration.splitlines()
    ), "calibration must return to the SR-OPD parent launcher"


def test_pda_launcher_separates_calibration_and_supports_two_step_smoke():
    script_dir = REPO_ROOT / "bash" / "sr_opd"
    base_config = (script_dir / "common" / "base_config.sh").read_text(encoding="utf-8")
    pda = (script_dir / "main" / "run_sr_opd_m32.sh").read_text(encoding="utf-8")
    train = (REPO_ROOT / "bash" / "train" / "sr-opd.sh").read_text(encoding="utf-8")

    assert "$OPD_ROOT/calibration/artifacts" in base_config
    assert "Calibration artifact ready:" in base_config
    assert "SR_OPD_SMOKE_STEPS" in pda
    assert "FF_FRESH_STEP_LIMIT" in pda
    assert "sr-opd-m32-seed" in pda
    assert "extra_student_forward=0" in pda
    assert "ff_opd.fresh_step_limit=${FF_FRESH_STEP_LIMIT:-0}" in train
    assert "skipping checkpoint post-processing" in train


def test_ablation_grid_stays_configurable():
    for num_boundaries in (2, 4, 8, 16, 32):
        assert BoundaryOPDSettings(num_boundaries=num_boundaries).num_boundaries == num_boundaries


def test_repository_never_requests_all_hidden_states_for_boundary_capture():
    boundary_source = Path(REPO_ROOT / "verl" / "verl" / "utils" / "boundary_opd.py").read_text(encoding="utf-8")
    assert "output_hidden_states" not in boundary_source
