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
    ('grid', 'scalar', 'action', 'next_grid', 'next_scalar', 'next_action_mask', 'reward')
)

# Hyperparameters
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
GAMMA = 0.98
LR = 1e-4
TARGET_UPDATE = 8192  # gradient steps
MIN_REPLAY_SIZE = 5000
TRAIN_EVERY_STEPS = 8
# Weight applied to the actual game score injected as a terminal reward bonus.
# Keeps it on the same scale as per-step rewards (max score ~15, max_steps=200).
SCORE_TERMINAL_WEIGHT = 0.1
ALL_COINS_BONUS = 2.0

END_OF_ROUND_OPT_STEPS = 10
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
ALL_COINS_TIME_BONUS_WEIGHT = 1.0 # Only used to determine best model

EPSILON_START = 1.0
EPSILON_END = 0.06
# For ~1000 games x ~200 steps = ~200k decisions:
# epsilon approaches its floor well before training ends.
EPSILON_HALF_LIFE = s.MAX_STEPS*300



from .Networks import DQN_prev as DQN_net

MAJOR_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.KILLED_SELF: -5.0,
    e.GOT_KILLED: -5.0,
    e.COIN_FOUND: 0.10,
    e.OPPONENT_ELIMINATED: 1.0,
    e.CRATE_DESTROYED: 0.6,
}

SHAPING_REWARDS = {
    e.REVERSED_DIRECTION: -0.05,
    e.MOVED_CLOSE_TO_ENEMY: 0.00,
    e.MOVED_AWAY_FROM_ENEMY: 0.00,
    e.IN_DANGER: -0.10,
    e.SAFE_WAIT: 0.00,
    e.WAITED: -0.05,
    e.INVALID_ACTION: 0.0,
    e.BOMB_DROPPED: -0.5,
    e.BOMB_EXPLODED: 0.0,
    e.SURVIVED_ROUND: 0.0,
}

def setup_training(self):
    """Initialise training-related objects for the agent."""

    self.replay_buffer = deque(maxlen=BUFFER_SIZE)
    self.steps_done = 0
    self.gradient_steps = 0
    self.round_coins_collected = 0
    self.round_number_kills = 0
    self.best_score = -1
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

    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR) #weight_decay=1e-4)

    self.loss_fn = nn.SmoothL1Loss()

    self.epsilon_start = EPSILON_START
    self.epsilon_end = EPSILON_END
    self.tau = EPSILON_HALF_LIFE


    def get_epsilon_value(agent):
        decay_steps = agent.steps_done
        return agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(
            -np.log(2.0) * decay_steps / agent.tau
        )

    self.get_epsilon = lambda: get_epsilon_value(self)

    
    self.logger.info(
        "Training initialized: buffer=%d, batch=%d, gamma=%.3f, lr=%g",
        BUFFER_SIZE,
        BATCH_SIZE,
        GAMMA,
        LR,
    )


def optimize_model(self):

    '''Perform double DQN optimization step on a batch of transitions from the replay buffer.'''

    if len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return None

    batch = random.sample(self.replay_buffer, BATCH_SIZE)

    states_grid_np = np.stack(
        [transition.grid for transition in batch],
        axis=0,
    ).astype(np.float32)

    states_scalar_np = np.stack(
        [transition.scalar for transition in batch],
        axis=0,
    ).astype(np.float32)

    # Store the action as an integer index.
    actions_np = np.array([ACTION_TO_INDEX[transition.action] for transition in batch], dtype=np.int64)

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
        ).astype(np.float32)

        next_scalars_np = np.stack(
            [transition.next_scalar for transition in non_final_transitions],
            axis=0,
        ).astype(np.float32)

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
        BATCH_SIZE,
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
    target_q = rewards + GAMMA * next_q

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
    if self.gradient_steps % TARGET_UPDATE == 0:
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
            reward += ALL_COINS_BONUS

            if getattr(self, "log_dqn_details", False):
                self.logger.info(
                    "All coins collected: completion bonus +%.2f",
                    ALL_COINS_BONUS,
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
                old_grid,
                old_scalar,
                self_action,
                new_grid,
                new_scalar,
                next_action_mask,
                reward,
            )
        )

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        self.steps_done += 1

        if self.steps_done % TRAIN_EVERY_STEPS == 0:
            optimize_model(self)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game to handle final transition and save model."""
    self._last_action_index = ACTION_TO_INDEX.get(last_action, ACTION_TO_INDEX["WAIT"])
    last_state = state_to_features_cached(self, last_game_state)
    reward = reward_from_events(self, events)

    # Inject actual game score as a terminal bonus so the Bellman target
    # is directly anchored to the real score, not only the step-reward proxy.
    actual_score = last_game_state["self"][1] if last_game_state is not None else 0
    terminal_bonus = float(actual_score) * SCORE_TERMINAL_WEIGHT

    reward += terminal_bonus
    if getattr(self, "log_dqn_details", False):
        self.logger.info(
            "End of round: actual_score=%d terminal_bonus=%.3f total_terminal_reward=%.3f",
            actual_score, terminal_bonus, reward,
        )

    # terminal state: next_state is None
    if last_state is not None:
        last_grid, last_scalar = last_state
        self.replay_buffer.append(Transition(last_grid, last_scalar, last_action, None, None, None, reward))

    # Do some final optimization passes
    for _ in range(END_OF_ROUND_OPT_STEPS):
        optimize_model(self)

    # Save latest policy network to MODEL_FILE; TRAINING_MODEL_FILE is the read-only start checkpoint
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    torch.save(self.policy_net.state_dict(), MODEL_FILE)

    current_score = float(self.round_coins_collected + self.round_number_kills * 5)
    if (
        last_game_state is not None
        and len(last_game_state.get("coins", [])) == 0
    ):
        time_left = max(0, int(s.MAX_STEPS) - int(last_game_state.get("step", 0))) # remaining steps in the round after last coin was collected
        time_left_fraction = float(time_left) / float(max(1, s.MAX_STEPS))
        current_score += ALL_COINS_TIME_BONUS_WEIGHT * time_left_fraction
        self.logger.info(
            "All coins collected with %d steps remaining (%.3f fraction of total steps).",
            time_left, time_left_fraction
        )
    if current_score > self.best_score:
        self.best_score = current_score
        torch.save(self.policy_net.state_dict(), BEST_MODEL_FILE)
        self.logger.info(
            "Saved new best model with score %.3f.",
            current_score,
        )

    self.round_coins_collected = 0
    self.round_number_kills = 0
    self.previous_old_position = None
    self.position_history.clear()


def reward_from_events(self, events: List[str]) -> float:
    """Map game events to scalar rewards.

    This function centralizes reward shaping. Values chosen below are a starting
    point and can be tuned. The function returns a float to allow fractional
    rewards (e.g., small penalties for dropping bombs).

    Design goals:
    - Encourage coin collection and finding coins.
    - Encourage destroying crates (makes coins available) but penalize suicides.
    - Penalize meaningless waiting or invalid actions.
    - Reward eliminating opponents and surviving rounds.
    """
    # Split rewards into:
    # - major outcomes (coin collection / death), which should dominate learning
    # - shaping terms (movement heuristics), which are clipped per step
    TIME_PENALTY = 0.01  # small penalty to encourage faster completion
    major_sum = 0.0
    shaping_sum = 0.0
    for event in events:
        major_sum += MAJOR_REWARDS.get(event, 0.0)
        shaping_sum += SHAPING_REWARDS.get(event, 0.0)

    # Prevent step-wise shaping terms from overshadowing major outcomes.
    shaping_sum = float(np.clip(shaping_sum, -0.3, 0.3))
    reward_sum = major_sum + shaping_sum - TIME_PENALTY

    if getattr(self, "log_dqn_details", False):
        self.logger.info(
            f"Awarded {reward_sum:.3f} (major={major_sum:.3f}, shaping={shaping_sum:.3f}) "
            f"for events {', '.join(events)}"
        )

    return float(reward_sum)
