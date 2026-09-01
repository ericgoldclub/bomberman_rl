import os
import random
from collections import deque, namedtuple
from typing import List
import copy
import atexit
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .callbacks import (
    ACTIONS,
    state_to_features_cached,
    features_used_for_action,
    LATEST_CHECKPOINT_FILE,
    BEST_MODEL_FILE,
    REPLAY_BUFFER_FILE,
    _feature_dimensions,
    _is_valid_action,
    _policy_action_mask,
)

import events as e
import settings as s

Transition = namedtuple(
    'Transition',
    ('grid', 'scalar', 'action_idx', 'next_grid', 'next_scalar', 'next_action_mask', 'reward')
)

ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
REQUIRED_HYPERPARAMETER_KEYS = (
    "BUFFER_SIZE",
    "BATCH_SIZE",
    "GAMMA",
    "LR",
    "TARGET_UPDATE",
    "MIN_REPLAY_SIZE",
    "TRAIN_EVERY_STEPS",
    "END_OF_ROUND_OPT_STEPS",
    "EPSILON_START",
    "EPSILON_END",
    "EPSILON_HALF_LIFE",
    "EVAL_EVERY_TRAINING_ROUNDS",
    "EVAL_ROUNDS",
)

HYPERPARAMS_FILE = os.path.join(os.path.dirname(__file__), "Hyperparams.prm")

CHECKPOINT_VERSION = 1
REPLAY_BUFFER_VERSION = 1
REPLAY_SAVE_EVERY_ROUNDS = 100

BEST_MODEL_METRIC_VERSION = 1
FEATURE_VERSION = 1
REWARD_VERSION = 1


from .Networks import DQN_prev as DQN_net

MAJOR_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.COIN_FOUND: 0.05,
    e.OPPONENT_ELIMINATED: 0.25,
    e.CRATE_DESTROYED: 0.15,
}

SHAPING_REWARDS = {
    e.REVERSED_DIRECTION: -0.015, # small penalty for reversing direction
    e.IN_DANGER: -0.03,
    e.WAITED: -0.01, # small penalty for waiting
}


TERMINAL_OBJECTIVE_ALIGNMENT_WEIGHT = 0.1 # multiplying the current score to shape the reward
ALL_COINS_CLEAR_BONUS = 5.0 # bonus reward for clearing all coins
STEP_TIME_COST = 0.005 # penalty for each step to encourage faster completion of objectives

def _transition_key(game_state: dict | None, action: str):
    """Identify the environment action represented by a replay transition."""
    if game_state is None:
        return None

    self_state = game_state.get("self")
    self_position = self_state[-1] if self_state is not None else None

    return (
        game_state.get("round"),
        game_state.get("step"),
        self_position,
        ACTION_TO_INDEX.get(action, ACTION_TO_INDEX["WAIT"]),
    )

def _pack_feature_array(arr: np.ndarray | None, dtype: np.dtype) -> np.ndarray | None:
    if arr is None:
        return None
    return np.asarray(arr, dtype=dtype)


def _pack_normalized_feature_array(arr: np.ndarray | None) -> np.ndarray | None:
    """Store normalized feature arrays as uint8 to reduce replay memory."""
    if arr is None:
        return None
    clipped = np.clip(arr, 0.0, 1.0)
    return np.rint(clipped * 255.0).astype(np.uint8)


def _unpack_normalized_feature_array(arr: np.ndarray) -> np.ndarray:
    """Restore uint8-packed normalized feature arrays to float32."""
    return arr.astype(np.float32) / 255.0


def _load_hyperparameters(self) -> None:
    """Load all training hyperparameters (and epsilon state) from Hyperparams.prm."""
    if not os.path.isfile(HYPERPARAMS_FILE):
        raise FileNotFoundError(f"Hyperparameter file not found: {HYPERPARAMS_FILE}")

    parsed: dict[str, float] = {}
    with open(HYPERPARAMS_FILE, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("#", 1)[0].strip().replace("_", "")
            if not value:
                continue
            parsed[key] = float(value)

    missing = [key for key in REQUIRED_HYPERPARAMETER_KEYS if key not in parsed]
    if missing:
        raise RuntimeError(
            f"Missing hyperparameter(s) in {HYPERPARAMS_FILE}: {', '.join(missing)}"
        )

    self.buffer_size = int(parsed["BUFFER_SIZE"])
    self.batch_size = int(parsed["BATCH_SIZE"])
    self.gamma = float(parsed["GAMMA"])
    self.lr = float(parsed["LR"])
    self.target_update = int(parsed["TARGET_UPDATE"])
    self.min_replay_size = int(parsed["MIN_REPLAY_SIZE"])
    self.train_every_steps = int(parsed["TRAIN_EVERY_STEPS"])
    self.end_of_round_opt_steps = int(parsed["END_OF_ROUND_OPT_STEPS"])
    self.epsilon_start = float(parsed["EPSILON_START"])
    self.epsilon_end = float(parsed["EPSILON_END"])
    self.tau = float(parsed["EPSILON_HALF_LIFE"])

    self.eval_every_training_rounds = int(parsed["EVAL_EVERY_TRAINING_ROUNDS"])
    self.eval_rounds = int(parsed["EVAL_ROUNDS"])

    self.epsilon_current = float(
        np.clip(
            parsed.get("EPSILON_LAST", self.epsilon_start),
            self.epsilon_end,
            self.epsilon_start,
        )
    )
    self.steps_done = int(parsed.get("STEPS_DONE", 0.0))

    if self.buffer_size <= 0:
        raise RuntimeError("BUFFER_SIZE must be > 0.")
    if self.batch_size <= 0:
        raise RuntimeError("BATCH_SIZE must be > 0.")
    if self.min_replay_size <= 0:
        raise RuntimeError("MIN_REPLAY_SIZE must be > 0.")
    if self.batch_size > self.buffer_size:
        raise RuntimeError(
            f"BATCH_SIZE ({self.batch_size}) must not exceed BUFFER_SIZE ({self.buffer_size})."
        )
    if self.eval_every_training_rounds <= 0:
        raise RuntimeError(
            "EVAL_EVERY_TRAINING_ROUNDS must be > 0."
        )

    if self.eval_rounds <= 0:
        raise RuntimeError("EVAL_ROUNDS must be > 0.")


def _save_hyperparameters(self) -> None:
    """Update runtime state fields in Hyperparams.prm without touching fixed parameters."""
    runtime_updates = {
        "EPSILON_LAST": f"{self.epsilon_current:.10f}",
        "STEPS_DONE": f"{int(self.steps_done)}",
        "BEST_MODEL_SCORE": f"{float(self.best_score):.1f}",
        "BEST_MODEL_COINS_COLLECTED": f"{int(self.best_score_coins_collected)}",
        "BEST_MODEL_ENEMIES_KILLED": f"{int(self.best_score_enemies_killed)}",
        "BEST_MODEL_TIME_LEFT_AFTER_ALL_COINS": f"{int(self.best_score_time_left_after_all_coins)}",
        "BEST_MODEL_COMPLETION_RATE": (f"{self.best_completion_rate:.5f}"),
        "BEST_MODEL_MEAN_TIME_LEFT": (f"{self.best_mean_time_left:.5f}"),
        "BEST_MODEL_METRIC_VERSION": (str(BEST_MODEL_METRIC_VERSION)),
    }

    lines: list[str] = []
    if os.path.isfile(HYPERPARAMS_FILE):
        with open(HYPERPARAMS_FILE, "r", encoding="utf-8") as fp:
            lines = fp.readlines()

    seen_keys = set()
    for idx, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue

        key = raw_line.split("=", 1)[0].strip()
        if key in runtime_updates:
            lines[idx] = f"{key}={runtime_updates[key]}\n"
            seen_keys.add(key)

    for key, value in runtime_updates.items():
        if key not in seen_keys:
            lines.append(f"{key}={value}\n")

    with open(HYPERPARAMS_FILE, "w", encoding="utf-8") as fp:
        fp.writelines(lines)

def _atomic_torch_save(payload, path: str) -> None:
    """Save a torch object without exposing a partially written file."""
    temporary_path = f"{path}.tmp.{os.getpid()}"

    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _save_latest_checkpoint(self) -> None:
    """Save everything required to resume the same training task."""
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "policy_state_dict": self.policy_net.state_dict(),
        "target_state_dict": self.target_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "steps_done": int(self.steps_done),
        "gradient_steps": int(self.gradient_steps),
        "epsilon_current": float(self.epsilon_current),
        "best_score": float(self.best_score),
        "best_score_coins_collected": int(
            self.best_score_coins_collected
        ),
        "best_score_enemies_killed": int(
            self.best_score_enemies_killed
        ),
        "best_score_time_left_after_all_coins": int(
            self.best_score_time_left_after_all_coins
        ),
        "grid_channels": int(self.grid_channels),
        "scalar_size": int(self.scalar_size),
        "actions": tuple(ACTIONS),
        "best_completion_rate": float(
            self.best_completion_rate
        ),
        "best_mean_time_left": float(
            self.best_mean_time_left
        ),
        "evaluation_round": bool(
            self._evaluation_round
        ),
        "evaluation_rounds_remaining": int(
            self._evaluation_rounds_remaining
        ),
        "evaluation_results": list(
            self._evaluation_results
        ),
        "training_rounds_since_evaluation": int(
            self._training_rounds_since_evaluation
        ),
        "best_model_metric_version": (
            BEST_MODEL_METRIC_VERSION
        ),

    }

    _atomic_torch_save(
        checkpoint,
        LATEST_CHECKPOINT_FILE,
    )

def _restore_latest_checkpoint(self) -> None:
    """Restore the non-policy state of a loaded latest checkpoint."""
    checkpoint = getattr(self, "_resume_checkpoint", None)

    if checkpoint is None:
        return

    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            "Unsupported checkpoint version: "
            f"{checkpoint.get('checkpoint_version')}"
        )

    if tuple(checkpoint.get("actions", ())) != tuple(ACTIONS):
        raise RuntimeError(
            "Checkpoint action ordering is incompatible."
        )

    if int(checkpoint.get("grid_channels", -1)) != int(
        self.grid_channels
    ):
        raise RuntimeError(
            "Checkpoint grid feature count is incompatible."
        )

    if int(checkpoint.get("scalar_size", -1)) != int(
        self.scalar_size
    ):
        raise RuntimeError(
            "Checkpoint scalar feature count is incompatible."
        )

    self.target_net.load_state_dict(
        checkpoint["target_state_dict"]
    )
    self.optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    self.steps_done = int(
        checkpoint.get("steps_done", self.steps_done)
    )
    self.gradient_steps = int(
        checkpoint.get("gradient_steps", self.gradient_steps)
    )
    self.epsilon_current = float(
        np.clip(
            checkpoint.get(
                "epsilon_current",
                self.epsilon_current,
            ),
            self.epsilon_end,
            self.epsilon_start,
        )
    )

    stored_metric_version = checkpoint.get(
        "best_model_metric_version"
    )

    if (
        stored_metric_version
        == BEST_MODEL_METRIC_VERSION
    ):
        self.best_score = float(
            checkpoint.get("best_score", -1.0)
        )
        self.best_score_coins_collected = int(
            checkpoint.get(
                "best_score_coins_collected",
                0,
            )
        )
        self.best_score_enemies_killed = int(
            checkpoint.get(
                "best_score_enemies_killed",
                0,
            )
        )
        self.best_score_time_left_after_all_coins = int(
            checkpoint.get(
                "best_score_time_left_after_all_coins",
                0,
            )
        )
        self.best_completion_rate = float(
            checkpoint.get(
                "best_completion_rate",
                0.0,
            )
        )
        self.best_mean_time_left = float(
            checkpoint.get(
                "best_mean_time_left",
                0.0,
            )
        )

    else:
        # Do not compare results produced by different definitions of "best".
        self.logger.warning(
            "Resetting best-model metric: "
            "checkpoint version=%s current version=%s.",
            stored_metric_version,
            BEST_MODEL_METRIC_VERSION,
        )

        self.best_score = -1.0
        self.best_score_coins_collected = 0
        self.best_score_enemies_killed = 0
        self.best_score_time_left_after_all_coins = 0
        self.best_completion_rate = 0.0
        self.best_mean_time_left = 0.0





    self._evaluation_round = bool(
        checkpoint.get("evaluation_round", False)
    )
    self._evaluation_rounds_remaining = int(
        checkpoint.get(
            "evaluation_rounds_remaining",
            0,
        )
    )
    self._evaluation_results = list(
        checkpoint.get("evaluation_results", [])
    )
    self._training_rounds_since_evaluation = int(
        checkpoint.get(
            "training_rounds_since_evaluation",
            0,
        )
    )
def _save_replay_buffer(self) -> None:
    """Save replay memory and compatibility metadata atomically."""
    transitions = list(self.replay_buffer)

    if self._pending_transition_key is not None and transitions:
        # Its terminal status is not yet known.
        transitions.pop()

    payload = {
        "replay_version": REPLAY_BUFFER_VERSION,
        "feature_version": FEATURE_VERSION,
        "reward_version": REWARD_VERSION,
        "capacity": int(self.buffer_size),
        "grid_channels": int(self.grid_channels),
        "scalar_size": int(self.scalar_size),
        "actions": tuple(ACTIONS),
        "steps_done": int(self.steps_done),
        "transitions": transitions,
    }

    temporary_path = (
        f"{REPLAY_BUFFER_FILE}.tmp.{os.getpid()}"
    )

    try:
        with open(temporary_path, "wb") as replay_file:
            pickle.dump(
                payload,
                replay_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            replay_file.flush()
            os.fsync(replay_file.fileno())

        os.replace(
            temporary_path,
            REPLAY_BUFFER_FILE,
        )

    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    self.logger.info(
        "Saved %d replay transitions to %s.",
        len(transitions),
        REPLAY_BUFFER_FILE,
    )

def _load_replay_buffer(self) -> None:
    """Load replay memory when resuming the same training task."""
    if not os.path.isfile(REPLAY_BUFFER_FILE):
        self.logger.warning(
            "No replay buffer found at %s.",
            REPLAY_BUFFER_FILE,
        )
        return

    with open(REPLAY_BUFFER_FILE, "rb") as replay_file:
        payload = pickle.load(replay_file)

    if not isinstance(payload, dict):
        raise RuntimeError("Invalid replay-buffer file.")

    if payload.get("replay_version") != REPLAY_BUFFER_VERSION:
        raise RuntimeError(
            "Unsupported replay-buffer version: "
            f"{payload.get('replay_version')}"
        )
    if (
        payload.get("feature_version")
        != FEATURE_VERSION
    ):
        raise RuntimeError(
            "Replay-buffer feature version is incompatible: "
            f"stored={payload.get('feature_version')} "
            f"current={FEATURE_VERSION}."
        )

    if (
        payload.get("reward_version")
        != REWARD_VERSION
    ):
        raise RuntimeError(
            "Replay-buffer reward version is incompatible: "
            f"stored={payload.get('reward_version')} "
            f"current={REWARD_VERSION}."
        )

    if tuple(payload.get("actions", ())) != tuple(ACTIONS):
        raise RuntimeError(
            "Replay-buffer action ordering is incompatible."
        )

    if int(payload.get("grid_channels", -1)) != int(
        self.grid_channels
    ):
        raise RuntimeError(
            "Replay-buffer grid features are incompatible."
        )

    if int(payload.get("scalar_size", -1)) != int(
        self.scalar_size
    ):
        raise RuntimeError(
            "Replay-buffer scalar features are incompatible."
        )

    transitions = payload.get("transitions")

    if not isinstance(transitions, list):
        raise RuntimeError(
            "Replay-buffer transitions are invalid."
        )

    self.replay_buffer.extend(
        transitions[-self.buffer_size:]
    )

    replay_steps_done = int(payload.get("steps_done", 0))

    step_difference = abs( int(self.steps_done) - replay_steps_done)

    if step_difference > self.buffer_size:
        self.logger.warning(
            "Checkpoint and replay differ by %d environment steps "
            "(checkpoint=%d replay=%d).",
            step_difference,
            self.steps_done,
            replay_steps_done,
        )

    self.logger.info(
        "Loaded %d replay transitions from %s.",
        len(self.replay_buffer),
        REPLAY_BUFFER_FILE,
    )

def _save_training_at_exit(self) -> None:
    """Best-effort save during a normal interpreter shutdown."""
    try:
        _save_latest_checkpoint(self)
        _save_replay_buffer(self)
        _save_hyperparameters(self)
    except Exception:
        self.logger.exception(
            "Could not save training state during shutdown."
        )

def _advance_epsilon(self) -> None:
    """Advance epsilon in memory by one environment step."""
    decay_factor = float(np.exp(-np.log(2.0) / self.tau) )

    self.epsilon_current = float(self.epsilon_end + (self.epsilon_current - self.epsilon_end)* decay_factor)

def _complete_evaluation_block(self) -> None:
    """Evaluate the candidate policy over the completed greedy rounds."""
    if not self._evaluation_results:
        raise RuntimeError(
            "Cannot complete an empty evaluation block."
        )

    game_scores = np.asarray(
        [
            result["game_score"]
            for result in self._evaluation_results
        ],
        dtype=np.float64,
    )

    completions = np.asarray(
        [
            result["completed"]
            for result in self._evaluation_results
        ],
        dtype=np.float64,
    )

    completed_times = [
        result["time_left"]
        for result in self._evaluation_results
        if result["completed"]
    ]

    mean_game_score = float(np.mean(game_scores))
    completion_rate = float(np.mean(completions))

    if completed_times:
        mean_time_left = float(
            np.mean(completed_times)
        )
    else:
        mean_time_left = 0.0

    candidate_metric = (
        mean_game_score,
        completion_rate,
        mean_time_left,
    )

    best_metric = (
        float(self.best_score),
        float(self.best_completion_rate),
        float(self.best_mean_time_left),
    )

    self.logger.info(
        "Greedy evaluation completed: "
        "mean_game_score=%.3f completion_rate=%.3f "
        "mean_time_left=%.1f over %d rounds.",
        mean_game_score,
        completion_rate,
        mean_time_left,
        len(self._evaluation_results),
    )

    if candidate_metric > best_metric:
        self.best_score = mean_game_score
        self.best_completion_rate = completion_rate
        self.best_mean_time_left = mean_time_left

        # These legacy fields remain available for logging and
        # Hyperparams.prm compatibility.
        self.best_score_coins_collected = int(
            round(mean_game_score)
        )
        self.best_score_time_left_after_all_coins = int(
            round(mean_time_left)
        )

        _atomic_torch_save(
            self.policy_net.state_dict(),
            BEST_MODEL_FILE,
        )

        self.logger.info(
            "Saved new best model: metric=%s.",
            candidate_metric,
        )

    else:
        self.logger.info(
            "Candidate did not improve best metric %s.",
            best_metric,
        )

    self._evaluation_results = []

def setup_training(self):
    """Initialise training-related objects for the agent."""

    _load_hyperparameters(self)
    training_start_mode = getattr(self, "training_start_mode", "resume",)

    self.replay_buffer = deque(maxlen=self.buffer_size)
    self.replay_start_size = min(
        self.buffer_size,
        max(self.min_replay_size, self.batch_size),
    )

    self.gradient_steps = 0
    self.round_coins_collected = 0
    self.round_number_kills = 0
    self.round_game_score = 0.0
    self.round_all_coins_collected = False

    # Define safe defaults before checkpoint restoration. In resume mode,
    # _restore_latest_checkpoint() replaces these values with the saved ones.
    self.best_score = -1.0
    self.best_score_coins_collected = 0
    self.best_score_enemies_killed = 0
    self.best_score_time_left_after_all_coins = 0

    if training_start_mode != "resume":
        # Fresh and transfer runs start a new training history.
        self.epsilon_current = self.epsilon_start
        self.steps_done = 0



    self.round_time_left_after_all_coins = 0
    self.previous_old_position = None
    self.previous_velocity = (0,0)
    self.position_history = deque(maxlen=4)
    self._last_action_index = ACTION_TO_INDEX["WAIT"]

    self._pending_transition_key = None

    self._evaluation_round = False
    self._evaluation_rounds_remaining = 0
    self._evaluation_results = []
    self._training_rounds_since_evaluation = 0

    self.best_completion_rate = 0.0
    self.best_mean_time_left = 0.0


    if not hasattr(self, "policy_net"):
        raise RuntimeError("policy_net must be initialized in setup() before calling setup_training()")

    self.policy_net.to(self.device)

    self.target_net = copy.deepcopy(self.policy_net)
    self.target_net.to(self.device)
    self.target_net.eval()  # target net is not trained, only used for evaluation
    for param in self.target_net.parameters():
        param.requires_grad = False 

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr)

    if training_start_mode == "resume":
        _restore_latest_checkpoint(self)
        _load_replay_buffer(self)

    self.loss_fn = nn.SmoothL1Loss()


    def get_epsilon_value(agent):
        return float(agent.epsilon_current)

    self.get_epsilon = lambda: get_epsilon_value(self)
    _save_hyperparameters(self)

    self._rounds_since_replay_save = 0

    if not getattr(
        self,
        "_training_exit_handler_registered",
        False,
    ):
        atexit.register(_save_training_at_exit, self)
        self._training_exit_handler_registered = True

    
    self.logger.info(
        "Training initialized: mode=%s buffer=%d replay=%d "
        "replay_start=%d batch=%d gamma=%.3f lr=%g "
        "steps=%d epsilon=%.5f",
        training_start_mode,
        self.buffer_size,
        len(self.replay_buffer),
        self.replay_start_size,
        self.batch_size,
        self.gamma,
        self.lr,
        self.steps_done,
        self.epsilon_current,
    )


def optimize_model(self):

    '''Perform double DQN optimization step on a batch of transitions from the replay buffer.'''

    replay_population = list(self.replay_buffer)

    # The latest transition remains in the replay buffer to ensure it is not lost or end_of_round() marks it terminal

    if self._pending_transition_key is not None and replay_population:
        replay_population.pop()

    if len(replay_population) < self.replay_start_size:
        return None

    batch = random.sample(replay_population, self.batch_size)




    states_grid_np = np.stack(
        [transition.grid for transition in batch],
        axis=0,
    )
    states_grid_np = _unpack_normalized_feature_array(states_grid_np)

    states_scalar_np = np.stack(
        [transition.scalar for transition in batch],
        axis=0,
    )
    states_scalar_np = _unpack_normalized_feature_array(states_scalar_np)

    # Store the action as an integer index.
    actions_np = np.array([transition.action_idx for transition in batch], dtype=np.int64)

    rewards_np = np.array(
        [transition.reward for transition in batch],
        dtype=np.float32,
    )

    # A transition is non-terminal if it contains a next state.
    non_final_mask_np = np.array(
        [transition.next_grid is not None for transition in batch],
        dtype=bool,
    )

    # Extract only non-terminal next states.
    non_final_transitions = [
        transition
        for transition in batch
        if transition.next_grid is not None
    ]

    if non_final_transitions:
        next_grids_np = np.stack(
            [transition.next_grid for transition in non_final_transitions],
            axis=0,
        )
        next_grids_np = _unpack_normalized_feature_array(next_grids_np)

        next_scalars_np = np.stack(
            [transition.next_scalar for transition in non_final_transitions],
            axis=0,
        )
        next_scalars_np = _unpack_normalized_feature_array(next_scalars_np)

        next_action_masks_np = np.stack(
            [
                transition.next_action_mask
                for transition in non_final_transitions
            ],
            axis=0,
        ).astype(bool)

    else:
        next_grids_np = None
        next_scalars_np = None
        next_action_masks_np = None



    states_grid = torch.from_numpy(states_grid_np).to(self.device)
    states_scalar = torch.from_numpy(states_scalar_np).to(self.device)

    actions = (
        torch.from_numpy(actions_np)
        .to(self.device)
        .unsqueeze(1)
    )

    rewards = (
        torch.from_numpy(rewards_np)
        .to(self.device)
        .unsqueeze(1)
    )

    current_q_values = self.policy_net(
        states_grid,
        states_scalar,
    )

    # Select Q-value corresponding to the action that was actually taken.
    current_q = current_q_values.gather(
        1,
        actions,
    )

    #Compute Double-DQN target
    #a* = argmax_a Q_online(s', a)
    #target = r + gamma * Q_target(s', a*)



    next_q = torch.zeros(
        self.batch_size,
        1,
        device=self.device,
    )

    if non_final_transitions:

        next_grids = torch.from_numpy(
            next_grids_np
        ).to(self.device)

        next_scalars = torch.from_numpy(
            next_scalars_np
        ).to(self.device)

        next_action_masks = torch.from_numpy(
            next_action_masks_np
        ).to(self.device)

        # Sanity check: every non-terminal state must have
        # at least one legal action.
        if not next_action_masks.any(dim=1).all():
            raise RuntimeError(
                "Found a non-terminal state with no valid actions."
            )

        with torch.no_grad():

            next_online_q_values = self.policy_net(
                next_grids,
                next_scalars,
            )

            # Prevent illegal actions from being selected.
            next_online_q_values = next_online_q_values.masked_fill(
                ~next_action_masks,
                float("-inf"),
            )

            next_actions = next_online_q_values.argmax(
                dim=1,
                keepdim=True,
            )

            next_target_q_values = self.target_net(
                next_grids,
                next_scalars,
            )

            next_target_q = next_target_q_values.gather(
                1,
                next_actions,
            )

        # Map the non-terminal next-Q values back to their
        # corresponding positions in the complete batch.
        non_final_mask = torch.from_numpy(
            non_final_mask_np
        ).to(self.device)

        next_q[non_final_mask] = next_target_q



    #Bellman target
    target_q = rewards + self.gamma * next_q

    # target_q is treated as a constant target.
    target_q = target_q.detach()


    loss = self.loss_fn(
        current_q,
        target_q,
    )

    self.optimizer.zero_grad(set_to_none=True)

    loss.backward()

    nn.utils.clip_grad_norm_(
        self.policy_net.parameters(),
        max_norm=5.0,
    )

    self.optimizer.step()

    self.gradient_steps += 1

    # Periodically synchronize target network.
    if self.gradient_steps % self.target_update == 0:
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        if getattr(self, "log_dqn_details", False):
            self.logger.info(
                "Target network updated at gradient step %d",
                self.gradient_steps,
            )


    if getattr(self, "log_dqn_details", False):
        self.logger.info(
            "loss=%.6f | gradient_step=%d | "
            "mean_Q=%.4f | mean_target=%.4f",
            loss.item(),
            self.gradient_steps,
            current_q.mean().item(),
            target_q.mean().item(),
        )

    return loss.item()

def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """Called once per step to calculate reward and store transition."""

    if self.logger.isEnabledFor(10):
        self.logger.debug(
            'Encountered game event(s) %s in step %s',
            ", ".join(map(repr, events)),
            new_game_state["step"],
        )

    events = list(events)

    if new_game_state is not None:
        self.round_game_score = float(new_game_state["self"][1])

    # ------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------

    coins_collected = events.count(e.COIN_COLLECTED)
    kills = events.count(e.KILLED_OPPONENT)

    self.round_coins_collected += coins_collected
    self.round_number_kills += kills

    # ------------------------------------------------------------
    # Additional event detection
    # ------------------------------------------------------------

    if new_game_state is not None:

        field = new_game_state["field"]

        explosion_map = new_game_state.get(
            "explosion_map",
            np.zeros_like(field),
        )

        _, _, _, (x, y) = new_game_state["self"]

        # Player currently standing in an explosion.
        if explosion_map[x, y] > 0:
            events.append(e.IN_DANGER)

        # --------------------------------------------------------
        # Movement-based shaping
        # --------------------------------------------------------

        if old_game_state is not None:

            old_pos = old_game_state["self"][-1]
            new_pos = new_game_state["self"][-1]

            current_velocity = (
                new_pos[0] - old_pos[0],
                new_pos[1] - old_pos[1],
            )

            previous_vx, previous_vy = self.previous_velocity
            current_vx, current_vy = current_velocity

            # Detect movement reversal.
            if (
                (current_vx != 0 or current_vy != 0)
                and
                (previous_vx != 0 or previous_vy != 0)
            ):
                dot_product = (
                    current_vx * previous_vx
                    + current_vy * previous_vy
                )

                if dot_product < 0:
                    events.append(e.REVERSED_DIRECTION)

            self.previous_old_position = old_pos
            self.previous_velocity = current_velocity

    # ------------------------------------------------------------
    # Calculate reward
    # ------------------------------------------------------------

    reward = reward_from_events(self, events)

    # ------------------------------------------------------------
    # Detect completion
    #
    # Give the bonus exactly when the last coin is collected.
    # ------------------------------------------------------------

    if (
        old_game_state is not None
        and new_game_state is not None
    ):
        old_coins = len(old_game_state["coins"])
        new_coins = len(new_game_state["coins"])

        if old_coins > 0 and new_coins == 0:
            time_left = max(0, int(s.MAX_STEPS) - int(new_game_state.get("step", 0)))
            self.round_all_coins_collected = True
            self.round_time_left_after_all_coins = time_left

            time_left_fraction = float(time_left) / float(max(1, s.MAX_STEPS))

            reward += (ALL_COINS_CLEAR_BONUS + time_left_fraction)

            if getattr(self, "log_dqn_details", False):
                self.logger.info(
                    "All coins collected: completion bonus +%.2f, "
                    "time bonus +%.3f (%d steps left)",
                    ALL_COINS_CLEAR_BONUS,
                    time_left_fraction,
                    time_left,
                )

    # ------------------------------------------------------------
    # Convert states into neural-network features
    # ------------------------------------------------------------

    if self._evaluation_round:
        # Evaluation measures the frozen greedy policy. It must not alter
        # replay, epsilon, the optimizer, or the target network.
        return

    old_feats = features_used_for_action(self, old_game_state)
    self._last_action_index = ACTION_TO_INDEX.get(self_action, ACTION_TO_INDEX["WAIT"])
    new_feats = state_to_features_cached(self, new_game_state)

    if old_feats is not None:

        old_grid, old_scalar = old_feats

        if new_feats is not None:
            new_grid, new_scalar = new_feats
            next_action_mask = _policy_action_mask(new_game_state)

        else:
            new_grid = None
            new_scalar = None
            next_action_mask = None

        # --------------------------------------------------------
        # Store transition
        # --------------------------------------------------------

        self.replay_buffer.append(
            Transition(
                _pack_normalized_feature_array(old_grid),
                _pack_normalized_feature_array(old_scalar),
                ACTION_TO_INDEX.get(self_action, ACTION_TO_INDEX["WAIT"]),
                _pack_normalized_feature_array(new_grid),
                _pack_normalized_feature_array(new_scalar),
                _pack_feature_array(next_action_mask, np.uint8),
                reward,
            )
        )

        self._pending_transition_key = _transition_key(old_game_state, self_action)

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        self.steps_done += 1
        _advance_epsilon(self)

        if self.steps_done % self.train_every_steps == 0:
            optimize_model(self)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game to handle final transition and save model."""
    if self._evaluation_round:
        events = list(events)

        # Dead agents do not receive game_events_occurred() for their
        # fatal action. Account for game-score events from that action.
        died_on_final_action = (
            e.GOT_KILLED in events
            or e.KILLED_SELF in events
        )

        if died_on_final_action and last_game_state is not None:
            score_before_final_action = float(
                last_game_state["self"][1]
            )

            final_score_delta = (
                events.count(e.COIN_COLLECTED)
                * float(s.REWARD_COIN)
                + events.count(e.KILLED_OPPONENT)
                * float(s.REWARD_KILL)
            )

            self.round_game_score = (
                score_before_final_action
                + final_score_delta
            )

        self._evaluation_results.append(
            {
                "game_score": float(
                    self.round_game_score
                ),
                "completed": bool(
                    self.round_all_coins_collected
                ),
                "time_left": int(
                    self.round_time_left_after_all_coins
                ),
            }
        )

        self._evaluation_rounds_remaining -= 1

        self.logger.info(
            "Evaluation round finished: score=%.1f "
            "completed=%s time_left=%d remaining=%d.",
            self.round_game_score,
            self.round_all_coins_collected,
            self.round_time_left_after_all_coins,
            self._evaluation_rounds_remaining,
        )

        if self._evaluation_rounds_remaining <= 0:
            _complete_evaluation_block(self)
            self._evaluation_round = False
            self._training_rounds_since_evaluation = 0

        self.round_game_score = 0.0
        self.round_coins_collected = 0
        self.round_number_kills = 0
        self.round_time_left_after_all_coins = 0
        self.round_all_coins_collected = False
        self.previous_old_position = None
        self.previous_velocity = (0, 0)
        self.position_history.clear()
        self._pending_transition_key = None

        _save_latest_checkpoint(self)
        _save_hyperparameters(self)
        return

    last_state = features_used_for_action(self, last_game_state)
    self._last_action_index = ACTION_TO_INDEX.get(last_action, ACTION_TO_INDEX["WAIT"])

    terminal_key = _transition_key(last_game_state, last_action)


    terminal_already_stored = (
        self._pending_transition_key == terminal_key
        and bool(self.replay_buffer)
    )

    if terminal_already_stored:
        # A surviving agent's final action was already stored by
        # game_events_occurred(). Preserve its immediate reward and only
        # remove the next state so the terminal target cannot bootstrap.
        transition = self.replay_buffer[-1]
        self.replay_buffer[-1] = transition._replace(
            next_grid=None,
            next_scalar=None,
            next_action_mask=None,
        )
    else:
        # Dead agents do not receive game_events_occurred() for the fatal
        # action, so their final transition must be created here.
        reward = reward_from_events(self, events)

        if last_state is not None:
            last_grid, last_scalar = last_state
            self.replay_buffer.append(
                Transition(
                    _pack_normalized_feature_array(last_grid),
                    _pack_normalized_feature_array(last_scalar),
                    ACTION_TO_INDEX.get(last_action, ACTION_TO_INDEX["WAIT"]),
                    None,
                    None,
                    None,
                    reward,
                )
            )

            # This action was not counted in game_events_occurred().
            self.steps_done += 1
            _advance_epsilon(self)

    # The final transition is now complete and can be sampled by the
    # end-of-round optimization passes.
    self._pending_transition_key = None

    # Do some final optimization passes
    for _ in range(self.end_of_round_opt_steps):
        optimize_model(self)


    self._training_rounds_since_evaluation += 1

    if (
        self._training_rounds_since_evaluation
        >= self.eval_every_training_rounds
    ):
        self._evaluation_round = True
        self._evaluation_rounds_remaining = (
            self.eval_rounds
        )
        self._evaluation_results = []

        self.logger.info(
            "Starting %d greedy evaluation rounds "
            "after %d training rounds.",
            self.eval_rounds,
            self._training_rounds_since_evaluation,
        )


    # Save the complete state required to resume this training task.
    _save_latest_checkpoint(self)

    self._rounds_since_replay_save += 1

    if (
        self._rounds_since_replay_save
        >= REPLAY_SAVE_EVERY_ROUNDS
    ):
        _save_replay_buffer(self)
        self._rounds_since_replay_save = 0

    if self.round_time_left_after_all_coins > 0:
        time_left_fraction = (
            float(self.round_time_left_after_all_coins)
            / float(max(1, s.MAX_STEPS))
        )

        self.logger.info(
            "All coins collected with %d steps remaining "
            "(%.3f fraction of total steps).",
            self.round_time_left_after_all_coins,
            time_left_fraction,
        )

    # Every round must start with clean episode statistics, regardless of
    # whether all coins were collected.
    self.round_game_score = 0.0
    self.round_coins_collected = 0
    self.round_number_kills = 0
    self.round_time_left_after_all_coins = 0
    self.round_all_coins_collected = False

    self.previous_old_position = None
    self.previous_velocity = (0, 0)
    self.position_history.clear()

    _save_hyperparameters(self)


def reward_from_events(self, events: List[str]) -> float:
    """Map game events to scalar rewards.

    This function centralizes reward shaping. Values chosen below are a starting
    point and can be tuned. The function returns a float to allow fractional
    rewards (e.g., small penalties for dropping bombs).

    Design goals:
    - Keep primary objective dominant: coins, kills, time efficiency.
    - Keep shaping strictly auxiliary and bounded.
    """
    # Split rewards into:
    # - major outcomes (coin collection / death), which should dominate learning
    # - shaping terms (movement heuristics), which are clipped per step
    major_sum = 0.0
    shaping_sum = 0.0
    for event in events:
        major_sum += MAJOR_REWARDS.get(event, 0.0)
        shaping_sum += SHAPING_REWARDS.get(event, 0.0)

    # Keep shaping auxiliary and apply one direct dense time pressure term.
    shaping_sum = float(np.clip(shaping_sum, -0.1, 0.1))
    reward_sum = major_sum + shaping_sum - STEP_TIME_COST

    if getattr(self, "log_dqn_details", False):
        self.logger.info(
            f"Awarded {reward_sum:.3f} (major={major_sum:.3f}, shaping={shaping_sum:.3f}) "
            f"for events {', '.join(events)}"
        )

    return float(reward_sum)
