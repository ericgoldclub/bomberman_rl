#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="${PROJECT_ROOT}/agent_code/dqn_agent_v3"
HYPERPARAMS_FILE="${AGENT_DIR}/Hyperparams.prm"

CURRENT_BEST_MODEL="${AGENT_DIR}/dqn-current-best-model.pt"
CURRENT_CHECKPOINT="${AGENT_DIR}/dqn-latest-checkpoint.pt"
CURRENT_BUFFER="${AGENT_DIR}/dqn-current-buffer.pkl"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_EXECUTABLE="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_EXECUTABLE="${PROJECT_ROOT}/.venv/bin/python"
elif [[ -x "${PROJECT_ROOT}/../.venv/bin/python" ]]; then
    PYTHON_EXECUTABLE="${PROJECT_ROOT}/../.venv/bin/python"
else
    PYTHON_EXECUTABLE="$(command -v python3)"
fi

if [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
    echo "Python executable is unavailable: ${PYTHON_EXECUTABLE}" >&2
    exit 1
fi

set_hyperparameter() {
    local key="$1"
    local value="$2"

    if ! grep -qE "^[[:space:]]*${key}[[:space:]]*=" "${HYPERPARAMS_FILE}"; then
        echo "Missing ${key} in ${HYPERPARAMS_FILE}" >&2
        exit 1
    fi

    sed -i -E \
        "s|^[[:space:]]*${key}[[:space:]]*=.*$|${key}=${value}|" \
        "${HYPERPARAMS_FILE}"
}

require_file() {
    local path="$1"
    if [[ ! -s "${path}" ]]; then
        echo "Required file does not exist or is empty: ${path}" >&2
        exit 1
    fi
}

require_new_output_path() {
    local path="$1"

    if [[ -e "${path}" ]]; then
        echo "Refusing to overwrite existing named artifact: ${path}" >&2
        echo "Choose a new output name or archive the existing file first." >&2
        exit 1
    fi
}

require_fresh_file() {
    local path="$1"
    local marker="$2"

    require_file "${path}"
    if [[ ! "${path}" -nt "${marker}" ]]; then
        echo "Training did not freshly update expected artifact: ${path}" >&2
        exit 1
    fi
}

cd -- "${PROJECT_ROOT}"

###############################################################################

echo "Starting fresh training stage for the killer curriculum."
NAME_OF_NEW_BEST_MODEL="${AGENT_DIR}/natural-killer.pt"
NAME_OF_NEW_BUFFER="${AGENT_DIR}/natural-killer-buffer.pkl"
HYPERPARAMS_SNAPSHOT="${AGENT_DIR}/natural-killer-Hyperparams.prm"
LOG_DIRECTORY="${AGENT_DIR}/logs/natural-killer"

# Validate everything before starting a potentially long training session.
require_new_output_path "${NAME_OF_NEW_BEST_MODEL}"
require_new_output_path "${NAME_OF_NEW_BUFFER}"
require_new_output_path "${HYPERPARAMS_SNAPSHOT}"

mkdir -p "${LOG_DIRECTORY}"

# This marker distinguishes files produced by this stage from stale current
# artifacts left by an older or interrupted run.
stage_marker="$(mktemp "${AGENT_DIR}/.training-stage-start.XXXXXX")"
trap 'rm -f -- "${stage_marker}"' EXIT

set_hyperparameter "GAMMA" "0.99"
set_hyperparameter "LR" "2.5e-4"
set_hyperparameter "TARGET_UPDATE" "1_000"
set_hyperparameter "MIN_REPLAY_SIZE" "2_000"
set_hyperparameter "TRAIN_EVERY_STEPS" "4"
set_hyperparameter "EPSILON_START" "0.80"
set_hyperparameter "EPSILON_END" "0.05"
set_hyperparameter "EPSILON_HALF_LIFE" "120_000"
set_hyperparameter "EVAL_EVERY_TRAINING_ROUNDS" "250"
set_hyperparameter "EVAL_ROUNDS" "30"

DQN_TRAINING_MODE="fresh" \
DQN_REWARD_PROFILE="killer" \
DQN_TOTAL_COINS="9" \
"${PYTHON_EXECUTABLE}" main.py play \
    --agents dqn_agent_v3 peaceful_agent peaceful_agent LTD_ZERO_2 \
    --train 1 \
    --scenario loot-crate15 \
    --n-rounds 20_000 \
    --no-gui \
    --match-name dqn-v3-natural-killer \
    --log-dir "${LOG_DIRECTORY}"

require_fresh_file "${CURRENT_BEST_MODEL}" "${stage_marker}"
require_fresh_file "${CURRENT_CHECKPOINT}" "${stage_marker}"
require_fresh_file "${CURRENT_BUFFER}" "${stage_marker}"
rm -f -- "${stage_marker}"
trap - EXIT

# Copy, rather than move, so current artifacts and all inputs remain available.
cp -- "${CURRENT_BEST_MODEL}" "${NAME_OF_NEW_BEST_MODEL}"
cp -- "${CURRENT_BUFFER}" "${NAME_OF_NEW_BUFFER}"
cp -- "${HYPERPARAMS_FILE}" "${HYPERPARAMS_SNAPSHOT}"

echo "Training stage completed successfully."
echo "Model:           ${NAME_OF_NEW_BEST_MODEL}"
echo "Replay buffer:   ${NAME_OF_NEW_BUFFER}"
echo "Hyperparameters: ${HYPERPARAMS_SNAPSHOT}"

