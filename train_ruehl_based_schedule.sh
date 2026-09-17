#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="${PROJECT_ROOT}/agent_code/RUEHL_BASED_AGENT"
HYPERPARAMS_FILE="${AGENT_DIR}/Hyperparams.prm"

CURRENT_BEST="${AGENT_DIR}/z_best-model.pt"
CURRENT_CHECKPOINT="${AGENT_DIR}/z_latest-checkpoint.pt"
CURRENT_REPLAY="${AGENT_DIR}/z_replay.pkl"

KILLER_ROUNDS="${KILLER_ROUNDS:-30000}"
SELF_PLAY_ROUNDS="${SELF_PLAY_ROUNDS:-30000}"
EVAL_EVERY="${EVAL_EVERY:-180}"
EVAL_ROUNDS="${EVAL_ROUNDS:-20}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-35000}"
LOG_EVAL_EVENTS="${LOG_EVAL_EVENTS:-${DQN_LOG_EVAL_EVENTS:-1}}"

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

for value in "${KILLER_ROUNDS}" "${SELF_PLAY_ROUNDS}" "${EVAL_EVERY}" "${EVAL_ROUNDS}" "${REPLAY_BUFFER_SIZE}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Round and evaluation settings must be positive integers." >&2
        exit 1
    fi
done

if [[ ! "${LOG_EVAL_EVENTS}" =~ ^(0|1|false|true|no|yes|off|on)$ ]]; then
    echo "LOG_EVAL_EVENTS must be a boolean (0/1, false/true, no/yes, or off/on)." >&2
    exit 1
fi

set_hyperparameter() {
    local key="$1"
    local value="$2"
    if ! grep -qE "^${key}=" "${HYPERPARAMS_FILE}"; then
        echo "Missing ${key} in ${HYPERPARAMS_FILE}" >&2
        exit 1
    fi
    sed -i -E "s|^${key}=.*$|${key}=${value}|" "${HYPERPARAMS_FILE}"
}

require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Required file is missing or empty: $1" >&2
        exit 1
    fi
}

require_new_file() {
    if [[ -e "$1" ]]; then
        echo "Refusing to overwrite existing training artifact: $1" >&2
        exit 1
    fi
}

require_fresh_file() {
    require_file "$1"
    if [[ ! "$1" -nt "$2" ]]; then
        echo "Training did not update expected artifact: $1" >&2
        exit 1
    fi
}

snapshot_paths() {
    local name="$1"
    echo "${AGENT_DIR}/${name}.pt"
    echo "${AGENT_DIR}/${name}-replay.pkl"
    echo "${AGENT_DIR}/${name}-Hyperparams.prm"
}

# Check all output names before spending time on the first stage.
for stage in classic-killer classic-self-play; do
    while IFS= read -r output; do
        require_new_file "${output}"
    done < <(snapshot_paths "${stage}")
done

set_hyperparameter "GAMMA" "0.99"
# Transfer learning should refine the pretrained policy, not rapidly replace it.
set_hyperparameter "LR" "5e-5"
# Each transition contains two 7x17x17 NumPy arrays. A 100,000-item
# buffer used about 1.3 GiB merely to deserialize and caused this stage to be
# killed under system memory pressure. Keep enough varied experience without
# crowding out the game and optimizer.
set_hyperparameter "BUFFER_SIZE" "${REPLAY_BUFFER_SIZE}"
set_hyperparameter "TARGET_UPDATE" "1000"
set_hyperparameter "MIN_REPLAY_SIZE" "2000"
set_hyperparameter "TRAIN_EVERY_STEPS" "4"
set_hyperparameter "EPSILON_END" "0.05"
set_hyperparameter "EVAL_EVERY_TRAINING_ROUNDS" "${EVAL_EVERY}"
set_hyperparameter "EVAL_ROUNDS" "${EVAL_ROUNDS}"

run_stage() {
    local name="$1"
    local mode="$2"
    local profile="$3"
    local scenario="$4"
    local rounds="$5"
    local epsilon="$6"
    local half_life="$7"
    local total_coins="$8"
    local input_model="$9"
    shift 9
    local agents=(RUEHL_BASED_AGENT "$@")
    local model_output="${AGENT_DIR}/${name}.pt"
    local replay_output="${AGENT_DIR}/${name}-replay.pkl"
    local params_output="${AGENT_DIR}/${name}-Hyperparams.prm"
    local log_directory="${AGENT_DIR}/logs/${name}"
    local marker

    if [[ "${mode}" == "transfer" ]]; then
        require_file "${input_model}"
        if [[ "${input_model}" == "${model_output}" ]]; then
            echo "A training stage cannot use its own output as its input: ${input_model}" >&2
            exit 1
        fi
    fi

    set_hyperparameter "EPSILON_START" "${epsilon}"
    set_hyperparameter "EPSILON_HALF_LIFE" "${half_life}"
    mkdir -p "${log_directory}"
    marker="$(mktemp "${AGENT_DIR}/.training-stage-start.XXXXXX")"
    trap 'rm -f -- "${marker}"' EXIT

    echo
    echo "Starting ${name}: scenario=${scenario}, coins=${total_coins}, profile=${profile}, rounds=${rounds}"

    if [[ "${mode}" == "transfer" ]]; then
        DQN_TRAINING_MODE="transfer" \
        DQN_PRETRAINED_MODEL="${input_model}" \
        DQN_REWARD_PROFILE="${profile}" \
        DQN_TOTAL_COINS="${total_coins}" \
        DQN_LOG_EVAL_EVENTS="${LOG_EVAL_EVENTS}" \
        "${PYTHON_EXECUTABLE}" main.py play \
            --agents "${agents[@]}" \
            --train 1 \
            --scenario "${scenario}" \
            --n-rounds "${rounds}" \
            --no-gui \
            --match-name "ruehl-${name}" \
            --log-dir "${log_directory}"
    else
        DQN_TRAINING_MODE="fresh" \
        DQN_REWARD_PROFILE="${profile}" \
        DQN_TOTAL_COINS="${total_coins}" \
        DQN_LOG_EVAL_EVENTS="${LOG_EVAL_EVENTS}" \
        "${PYTHON_EXECUTABLE}" main.py play \
            --agents "${agents[@]}" \
            --train 1 \
            --scenario "${scenario}" \
            --n-rounds "${rounds}" \
            --no-gui \
            --match-name "ruehl-${name}" \
            --log-dir "${log_directory}"
    fi

    require_fresh_file "${CURRENT_BEST}" "${marker}"
    require_fresh_file "${CURRENT_CHECKPOINT}" "${marker}"
    require_fresh_file "${CURRENT_REPLAY}" "${marker}"
    cp -- "${CURRENT_BEST}" "${model_output}"
    cp -- "${CURRENT_REPLAY}" "${replay_output}"
    cp -- "${HYPERPARAMS_FILE}" "${params_output}"
    rm -f -- "${marker}"
    trap - EXIT

    echo "Finished ${name}. Best model: ${model_output}"
}

cd -- "${PROJECT_ROOT}"

run_stage \
    "classic-killer" "transfer" "killer" "classic" "${KILLER_ROUNDS}" \
    "0.25" "600_000" "9" "${AGENT_DIR}/loot-crate30-killer.pt" \
    rule_based_agent rule_based_agent rule_based_agent

# Continue in classic against a frozen copy of the current policy and two
# rule-based opponents. Only the first RUEHL_BASED_AGENT is trained.
run_stage \
    "classic-self-play" "transfer" "killer" "classic" "${SELF_PLAY_ROUNDS}" \
    "0.25" "600_000" "9" "${AGENT_DIR}/classic-killer.pt" \
    RUEHL_BASED_AGENT rule_based_agent rule_based_agent

echo
echo "RUEHL_BASED_AGENT curriculum completed."
echo "Final model: ${AGENT_DIR}/classic-self-play.pt"
