import inspect
from pathlib import Path

from verl.utils.ff_opd import FFOPDConfig
from verl.utils.ta_opd import compute_ta_opd_mask

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_full_opd_keeps_the_8192_response_horizon():
    train = (REPO_ROOT / "bash/train/full-opd.sh").read_text(encoding="utf-8")

    assert "MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}" in train


def test_main_ta_opd_experiments_fix_retention_ratio_to_point_10():
    launchers = (
        "run_group_a_qwen3_4b_to_qwen3_1.7b.sh",
        "run_group_c_skywork_or1_7b_to_deepseek_r1_1.5b.sh",
    )
    for launcher in launchers:
        source = (REPO_ROOT / "bash" / launcher).read_text(encoding="utf-8")
        assert 'TA_TOKEN_RETAIN_RATIO="${TA_TOKEN_RETAIN_RATIO:-0.10}"' in source
        assert "TA_TOKEN_RETAIN_RATIO=0.05" not in source


def test_all_opd_methods_share_the_5e_minus_6_actor_learning_rate():
    train = (REPO_ROOT / "bash/train/full-opd.sh").read_text(encoding="utf-8")
    actor = (
        REPO_ROOT / "verl/verl/trainer/config/actor/actor.yaml"
    ).read_text(encoding="utf-8")

    assert "actor_rollout_ref.actor.optim.lr=5e-6" in train
    assert "  lr: 5e-6" in actor


def test_ta_has_one_operational_default_retain_ratio():
    train = (REPO_ROOT / "bash/train/ta-opd.sh").read_text(encoding="utf-8")
    rollout_config = (
        REPO_ROOT / "verl/verl/workers/config/rollout.py"
    ).read_text(encoding="utf-8")

    assert "TOKEN_RETAIN_RATIO=0.10" in train
    assert "ta_opd_retain_ratio: float = 0.10" in rollout_config
    assert inspect.signature(compute_ta_opd_mask).parameters["retain_ratio"].default == 0.10
    assert "effective_token_retain_ratio=$TA_OPD_RETAIN_RATIO" in train


def test_all_baselines_use_sampled_token_loss_support():
    train = (REPO_ROOT / "bash/train/full-opd.sh").read_text(encoding="utf-8")
    tlr = (REPO_ROOT / "bash/train/tlr-opd.sh").read_text(encoding="utf-8")

    assert "else\n    export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}" in train
    assert 'LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-0}"' in tlr
    assert "export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}" in train


def test_ta_union_is_selector_only():
    train = (REPO_ROOT / "bash/train/ta-opd.sh").read_text(encoding="utf-8")
    trainer = (REPO_ROOT / "verl/verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")

    assert "TA_SCORE_TOP_K_STRATEGY=${TA_SCORE_TOP_K_STRATEGY:-union}" in train
    assert "ta_opd_score_top_k_strategy=$TA_SCORE_TOP_K_STRATEGY" in train
    assert 'batch.meta_info["selector_top_k"]' in trainer
    assert 'batch.meta_info["selector_top_k_strategy"]' in trainer


def test_ff_names_the_historical_cap_separately_from_realized_rate():
    config = FFOPDConfig()

    assert config.legacy_teacher_query_ratio == 0.25
    assert not hasattr(config, "teacher_query_ratio")


def test_ff_main_experiment_disables_retry_exposure_by_default():
    config = FFOPDConfig()
    launcher = (REPO_ROOT / "bash/train/sr-opd.sh").read_text(encoding="utf-8")

    assert config.max_no_success_retries == 0
    assert "MAX_NO_SUCCESS_RETRIES=0" in launcher
    assert "--max-no-success-retries" in launcher


def test_boundary_contrast_launcher_has_no_retry_phase_or_name_suffix():
    launcher = (REPO_ROOT / "bash/sr_opd/common/_run_sr_opd.sh").read_text(encoding="utf-8")

    assert "FF_MAX_NO_SUCCESS_RETRIES" not in launcher
    assert "retry0" not in launcher


def test_student_thinking_is_disabled_across_train_infer_and_eval():
    train = (REPO_ROOT / "bash/train/full-opd.sh").read_text(encoding="utf-8")
    infer = (REPO_ROOT / "scripts/infer/vllm_rollout.py").read_text(encoding="utf-8")
    evaluate = (REPO_ROOT / "scripts/val/eval/gen_vllm.py").read_text(encoding="utf-8")

    assert "export STUDENT_ENABLE_THINKING=False" in train
    assert "effective_student_enable_thinking=$STUDENT_ENABLE_THINKING" in train
    assert "apply_chat_template_kwargs.enable_thinking=$STUDENT_ENABLE_THINKING" in train
    for source in (infer, evaluate):
        assert "STUDENT_ENABLE_THINKING = False" in source
        assert "enable_thinking=STUDENT_ENABLE_THINKING" in source
        assert "Student thinking mode is fixed to false" in source
