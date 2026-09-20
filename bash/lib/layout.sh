#!/usr/bin/env bash
# Shared layout helpers for OPD bash entry points.
#
# Centralises the directory-naming conventions used by the train entry points, produce.sh,
# eval.sh, and the ablation sweeps so that every script agrees on where runs,
# suites, and evaluation artefacts live.
#
# Functions exposed:
#   opd_model_pair <teacher_name> <student_name>
#   opd_decimal_2 <value>
#   opd_timestamp
#   opd_method_dir <query_method> <post_method> [experiment_method]
#   opd_training_run_name <method_dir> <query_ratio> <exploration_rate>
#       <retain_ratio> <train_seed> <timestamp>
#   opd_mode_tag <query_ratio> <exploration_rate> <post_method> <retain_ratio>
#   opd_run_dir <output_root> <model_pair> <method> <run_id>
#   opd_suite_dir <output_root> <model_pair> <suite_name> <suite_id>
#   opd_component_dir <output_root> <component_name> <run_id>
#   opd_run_metadata_file <run_dir>
#   opd_eval_seed_tag <seed1> [seed2 ...]
#   opd_eval_checkpoint_name <step>

# Build the model-pair subdirectory name as <Teacher>__to__<Student>.
# Caller passes basenames so the result is a single directory component.
opd_model_pair() {
    local teacher_name="${1:?opd_model_pair: teacher name required}"
    local student_name="${2:?opd_model_pair: student name required}"
    printf '%s__to__%s\n' "$teacher_name" "$student_name"
}

# Format ratio-like values uniformly so equivalent configurations cannot create
# separate q0.4/q0.40 directories.
opd_decimal_2() {
    local value="${1:?opd_decimal_2: value required}"
    awk -v value="$value" 'BEGIN {
        if (value !~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)$/) {
            print "opd_decimal_2: invalid decimal: " value > "/dev/stderr"
            exit 2
        }
        printf "%.2f\n", value + 0
    }'
}

opd_timestamp() {
    date '+%Y%m%dT%H%M%S'
}

# Map implementation switches to the stable public method directory.
opd_method_dir() {
    local query_method="${1:?opd_method_dir: query method required}"
    local post_method="${2:-full}"
    local experiment_method="${3:-$query_method}"
    case "$experiment_method" in
        base) printf 'base\n'; return ;;
        teacher) printf 'teacher\n'; return ;;
    esac
    if [[ "$post_method" == "ta" ]]; then
        printf 'ta_opd\n'
        return
    fi
    case "$query_method" in
        full) printf 'full_opd\n' ;;
        *) echo "opd_method_dir: unsupported method: $query_method" >&2; return 2 ;;
    esac
}

opd_training_run_name() {
    local method_dir="${1:?opd_training_run_name: method required}"
    local query_ratio="${2:?opd_training_run_name: query ratio required}"
    local exploration_rate="${3:-0.0}"
    local retain_ratio="${4:-1.0}"
    local train_seed="${5:?opd_training_run_name: train seed required}"
    local timestamp="${6:?opd_training_run_name: timestamp required}"
    query_ratio="$(opd_decimal_2 "$query_ratio")" || return
    exploration_rate="$(opd_decimal_2 "$exploration_rate")" || return
    retain_ratio="$(opd_decimal_2 "$retain_ratio")" || return
    case "$method_dir" in
        full_opd)
            printf 'q%s-trainseed%s-%s\n' "$query_ratio" "$train_seed" "$timestamp" ;;
        ta_opd)
            printf 'q%s-r%s-trainseed%s-%s\n' "$query_ratio" "$retain_ratio" "$train_seed" "$timestamp" ;;
        *) echo "opd_training_run_name: unsupported method directory: $method_dir" >&2; return 2 ;;
    esac
}

opd_eval_run_name() {
    local timestamp="${1:?opd_eval_run_name: timestamp required}"
    printf 'evalseedset42-46-%s\n' "$timestamp"
}

# Normalise a decimal value so that integers get a trailing ".1" suffix
# (e.g. "1" -> "1.0") and fractional values have trailing zeros stripped.
# This keeps directory names stable regardless of how callers pass the value.
opd_normalize_decimal() {
    awk -v value="${1:?opd_normalize_decimal: value required}" 'BEGIN {
        number = value + 0
        if (number == int(number)) {
            printf "%.1f\n", number
        } else {
            text = sprintf("%.12f", number)
            sub(/0+$/, "", text)
            print text
        }
    }'
}

# Produce a compact human-readable tag describing the OPD configuration so
# that logs and metadata can surface it without parsing the full env.
opd_mode_tag() {
    local query_ratio="${1:?opd_mode_tag: query_ratio required}"
    local exploration_rate="${2:?opd_mode_tag: exploration_rate required}"
    local post_method="${3:?opd_mode_tag: post_method required}"
    local token_retain_ratio="${4:-1.0}"
    query_ratio="$(opd_normalize_decimal "$query_ratio")"
    exploration_rate="$(opd_normalize_decimal "$exploration_rate")"
    token_retain_ratio="$(opd_normalize_decimal "$token_retain_ratio")"
    printf 'query-%s-exploration-%s-post-method-%s-retain-%s\n' \
        "$query_ratio" "$exploration_rate" "$post_method" "$token_retain_ratio"
}

# Canonical run directory: <output_root>/<model_pair>/<method>/<run_id>
# All four arguments are required; callers are responsible for sanitising them.
opd_run_dir() {
    local output_root="${1:?opd_run_dir: output_root required}"
    local model_pair="${2:?opd_run_dir: model_pair required}"
    local method="${3:?opd_run_dir: method required}"
    local run_id="${4:?opd_run_dir: run_id required}"
    printf '%s/%s/%s/%s\n' "${output_root%/}" "$model_pair" "$method" "$run_id"
}

# Suite directory for grouped sweeps:
#   <output_root>/<model_pair>/_suites/<suite_name>/<suite_id>
opd_suite_dir() {
    local output_root="${1:?opd_suite_dir: output_root required}"
    local model_pair="${2:?opd_suite_dir: model_pair required}"
    local suite_name="${3:?opd_suite_dir: suite_name required}"
    local suite_id="${4:?opd_suite_dir: suite_id required}"
    printf '%s/%s/_suites/%s/%s\n' \
        "${output_root%/}" "$model_pair" "$suite_name" "$suite_id"
}

# Component-level ablation directory:
#   <output_root>/component-ablations/<component_name>/<run_id>
opd_component_dir() {
    local output_root="${1:?opd_component_dir: output_root required}"
    local component_name="${2:?opd_component_dir: component_name required}"
    local run_id="${3:?opd_component_dir: run_id required}"
    printf '%s/component-ablations/%s/%s\n' \
        "${output_root%/}" "$component_name" "$run_id"
}

# Path to the run.env metadata file written by the train entry points.
opd_run_metadata_file() {
    local run_dir="${1:?opd_run_metadata_file: run_dir required}"
    printf '%s/metadata/run.env\n' "${run_dir%/}"
}

# Evaluation seed tag. Callers may pass either individual seeds
# (`opd_eval_seed_tag 42 43 44 45 46` -> `seeds-42_43_44_45_46`) or a single
# underscore-joined token (`opd_eval_seed_tag 42_43_44_45_46`) which is the
# shape used by the ablation sweep scripts.
opd_eval_seed_tag() {
    [[ $# -gt 0 ]] || { echo "opd_eval_seed_tag: at least one seed required" >&2; return 1; }
    local joined="" seed
    for seed in "$@"; do
        [[ -z "$joined" ]] || joined+="_"
        joined+="$seed"
    done
    printf 'seeds-%s\n' "$joined"
}

# Stable public artifact name for a given internal global step.
opd_eval_checkpoint_name() {
    local step="${1:?opd_eval_checkpoint_name: step required}"
    printf 'step_%04d\n' "$step"
}
