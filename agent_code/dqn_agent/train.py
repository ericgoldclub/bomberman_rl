import os
import random
from collections import deque, namedtuple
from typing import List
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .callbacks import ACTIONS, state_to_features_cached, MODEL_FILE, BEST_MODEL_FILE, TRAINING_MODEL_FILE, _feature_dimensions, _is_valid_action, _policy_action_mask
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
)

HYPERPARAMS_FILE = os.path.join(os.path.dirname(__file__), "Hyperparams.prm")



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
POST_CLEAR_TIME_LEFT_WEIGHT = 1.0 # rewarding the time till end of game after all coins are cleared
STEP_TIME_COST = 0.1 # penalty for each step to encourage faster completion of objectives


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
    self.epsilon_current = float(
        np.clip(
            parsed.get("EPSILON_LAST", self.epsilon_start),
            self.epsilon_end,
            self.epsilon_start,
        )
    )
    self.steps_done = int(parsed.get("STEPS_DONE", 0.0))


def _save_hyperparameters(self) -> None:
    """Update runtime state fields in Hyperparams.prm without touching fixed parameters."""
    runtime_updates = {
        "EPSILON_LAST": f"{self.epsilon_current:.10f}",
        "STEPS_DONE": f"{int(self.steps_done)}",
        "BEST_MODEL_SCORE": f"{float(self.best_score):.1f}",
        "BEST_MODEL_COINS_COLLECTED": f"{int(self.best_score_coins_collected)}",
        "BEST_MODEL_ENEMIES_KILLED": f"{int(self.best_score_enemies_killed)}",
        "BEST_MODEL_TIME_LEFT_AFTER_ALL_COINS": f"{int(self.best_score_time_left_after_all_coins)}",
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


def _advance_and_persist_epsilon(self) -> None:
    """Advance epsilon by one environment step and persist to Hyperparams.prm."""
    decay_factor = float(np.exp(-np.log(2.0) / self.tau))
    self.epsilon_current = float(
        self.epsilon_end + (self.epsilon_current - self.epsilon_end) * decay_factor
    )
    _save_hyperparameters(self)


def _round_objective_score(self) -> float:
    """Primary round objective shared by training bonus and checkpoint selection."""
    time_left_fraction = float(self.round_time_left_after_all_coins) / float(max(1, s.MAX_STEPS))
    return float(
        self.round_coins_collected
        + self.round_number_kills * 5
        + POST_CLEAR_TIME_LEFT_WEIGHT * time_left_fraction
    )

def setup_training(self):
    """Initialise training-related objects for the agent."""

    _load_hyperparameters(self)

    self.replay_buffer = deque(maxlen=self.buffer_size)
    self.gradient_steps = 0
    self.round_coins_collected = 0
    self.round_number_kills = 0
    self.best_score = -1
    self.best_score_coins_collected = 0
    self.best_score_enemies_killed = 0
    self.best_score_time_left_after_all_coins = 0
    self.round_time_left_after_all_coins = 0
    self.previous_old_position = None
    self.previous_velocity = (0,0)
    self.position_history = deque(maxlen=4)
    self._last_action_index = ACTION_TO_INDEX["WAIT"]


    if not hasattr(self, "policy_net"):
        raise RuntimeError("policy_net must be initialized in setup() before calling setup_training()")

    self.policy_net.to(self.device)

    self.target_net = copy.deepcopy(self.policy_net)
    self.target_net.to(self.device)
    self.target_net.eval()  # target net is not trained, only used for evaluation
    for param in self.target_net.parameters():
        param.requires_grad = False 

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr) #weight_decay=1e-4)

    self.loss_fn = nn.SmoothL1Loss()


    def get_epsilon_value(agent):
        return float(agent.epsilon_current)

    self.get_epsilon = lambda: get_epsilon_value(self)
    _save_hyperparameters(self)

    
    self.logger.info(
        "Training initialized: buffer=%d, batch=%d, gamma=%.3f, lr=%g",
        self.buffer_size,
        self.batch_size,
        self.gamma,
        self.lr,
    )


def optimize_model(self):

    '''Perform double DQN optimization step on a batch of transitions from the replay buffer.'''

    if len(self.replay_buffer) < self.min_replay_size:
        return None

    batch = random.sample(self.replay_buffer, self.batch_size)

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
            self.round_time_left_after_all_coins = time_left
            reward += ALL_COINS_CLEAR_BONUS

            if getattr(self, "log_dqn_details", False):
                self.logger.info(
                    "All coins collected: completion bonus +%.2f (%d steps left)",
                    ALL_COINS_CLEAR_BONUS,
                    time_left,
                )

    # ------------------------------------------------------------
    # Convert states into neural-network features
    # ------------------------------------------------------------

    old_feats = state_to_features_cached(self, old_game_state)
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

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        self.steps_done += 1
        _advance_and_persist_epsilon(self)

        if self.steps_done % self.train_every_steps == 0:
            optimize_model(self)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game to handle final transition and save model."""
    self._last_action_index = ACTION_TO_INDEX.get(last_action, ACTION_TO_INDEX["WAIT"])
    last_state = state_to_features_cached(self, last_game_state)
    reward = reward_from_events(self, events)
    current_score = _round_objective_score(self)

    # Keep terminal bonus aligned with the exact best-model objective.
    terminal_bonus = current_score * TERMINAL_OBJECTIVE_ALIGNMENT_WEIGHT

    reward += terminal_bonus
    if getattr(self, "log_dqn_details", False):
        self.logger.info(
            "End of round: objective_score=%.3f terminal_bonus=%.3f total_terminal_reward=%.3f",
            current_score, terminal_bonus, reward,
        )

    # terminal state: next_state is None
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

    # Do some final optimization passes
    for _ in range(self.end_of_round_opt_steps):
        optimize_model(self)

    # Save latest policy network to MODEL_FILE; TRAINING_MODEL_FILE is the read-only start checkpoint
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    torch.save(self.policy_net.state_dict(), MODEL_FILE)

    if self.round_time_left_after_all_coins > 0:
        time_left_fraction = float(self.round_time_left_after_all_coins) / float(max(1, s.MAX_STEPS))
        self.logger.info(
            "All coins collected with %d steps remaining (%.3f fraction of total steps).",
            self.round_time_left_after_all_coins,
            time_left_fraction,
        )
    if current_score > self.best_score:
        self.best_score = current_score
        self.best_score_coins_collected = int(self.round_coins_collected)
        self.best_score_enemies_killed = int(self.round_number_kills)
        self.best_score_time_left_after_all_coins = int(self.round_time_left_after_all_coins)
        torch.save(self.policy_net.state_dict(), BEST_MODEL_FILE)
        self.logger.info(
            "Saved new best model with score %.3f.",
            current_score,
        )

    self.round_coins_collected = 0
    self.round_number_kills = 0
    self.round_time_left_after_all_coins = 0
    self.previous_old_position = None
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
