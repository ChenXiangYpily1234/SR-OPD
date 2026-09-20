#!/usr/bin/env bash

# Shared defaults for the paper's Boundary-Contrast ablations. Callers may
# override every value through the environment before launching a script.
export SEED="${SEED:-42}"
export FF_K_ROLLOUTS="${FF_K_ROLLOUTS:-4}"
export FF_SELECTOR_MODE="${FF_SELECTOR_MODE:-boundary_opd}"
export BOUNDARY_OPD_NUM_BOUNDARIES="${BOUNDARY_OPD_NUM_BOUNDARIES:-16}"
export BOUNDARY_OPD_SIMILARITY_METRIC="${BOUNDARY_OPD_SIMILARITY_METRIC:-centered_hidden_state_cosine}"
export BOUNDARY_OPD_FALLBACK_TO_FF_COST="${BOUNDARY_OPD_FALLBACK_TO_FF_COST:-false}"
export FF_SELECTOR_COST_AWARE="${FF_SELECTOR_COST_AWARE:-True}"
export BOUNDARY_METHOD_DIR="${BOUNDARY_METHOD_DIR:-sr_opd}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS="${BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS:-64}"
export BOUNDARY_OPD_CALIBRATION_SEED="${BOUNDARY_OPD_CALIBRATION_SEED:-42}"

boundary_calibration_manifest() {
    local num_boundaries="${1:?boundary count is required}"
    local calibration_root="${BOUNDARY_CALIBRATION_ROOT:-$OPD_ROOT/calibration/artifacts}"
    [[ "$calibration_root" = /* ]] || calibration_root="$OPD_ROOT/$calibration_root"
    local representation_domain="direct_hidden_state"
    printf '%s/%s/%s-m%s-seed%s-p%s/manifest.json\n' \
        "${calibration_root%/}" "$representation_domain" "$(basename "$ACTOR_MODEL_PATH")" "$num_boundaries" \
        "$BOUNDARY_OPD_CALIBRATION_SEED" "$BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS"
}

boundary_sha256_file() {
    local file_path="${1:?file path is required}"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file_path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file_path" | awk '{print $1}'
    else
        python3 -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$file_path"
    fi
}

boundary_ensure_calibration() {
    local num_boundaries="${1:?boundary count is required}"
    local manifest="${BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH:-}"
    local checksum

    if [[ -z "$manifest" ]]; then
        manifest="$(boundary_calibration_manifest "$num_boundaries")"
    elif [[ "$manifest" != /* ]]; then
        manifest="$OPD_ROOT/$manifest"
    fi

    if [[ ! -f "$manifest" ]]; then
        echo "==> Calibration artifact for M=$num_boundaries is missing; generating $manifest"
        local representation_domain="direct_hidden_state"
        BOUNDARY_OPD_NUM_BOUNDARIES="$num_boundaries" \
        BOUNDARY_OPD_CALIBRATION_REPRESENTATION_DOMAIN="$representation_domain" \
        BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH="$manifest" \
        BOUNDARY_OPD_CALIBRATION_ARTIFACT_SHA256= \
            bash "$OPD_ROOT/bash/sr_opd/representation/run_calibration.sh" "$num_boundaries"
    fi

    [[ -f "$manifest" ]] || {
        echo "ERROR: calibration run did not create $manifest" >&2
        return 2
    }
    checksum="$(boundary_sha256_file "$manifest")"
    [[ "$checksum" =~ ^[0-9a-f]{64}$ ]] || {
        echo "ERROR: failed to compute calibration manifest SHA-256: $manifest" >&2
        return 2
    }
    export BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH="$manifest"
    export BOUNDARY_OPD_CALIBRATION_ARTIFACT_SHA256="$checksum"
    echo "Calibration artifact ready: $manifest"
}
