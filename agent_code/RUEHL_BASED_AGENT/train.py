import os
import pickle
import random
from collections import Counter, deque, namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

import events as e
import settings as s
from .callbacks import (
    ACTIONS,
    BEST_MODEL_FILE,
    LATEST_CHECKPOINT_FILE,
    TOTAL_COINS,
    action_mask,
    blast_tiles,
    features_used_for_action,
    state_to_features,
)


DIRECTORY = os.path.dirname(__file__)
HYPERPARAMS_FILE = os.path.join(DIRECTORY, "Hyperparams.prm")
REPLAY_FILE = os.path.join(DIRECTORY, "replay.pkl")
MODEL_NAME = "ruehl-dqn-v1"
REPLAY_SAVE_EVERY_ROUNDS = 100

Transition = namedtuple(
    "Transition", "board scalar action next_board next_scalar next_mask reward"
)
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


# The four profiles are intentionally just reward tables. This makes it easy to
# change the training task without adding another training loop.
BASE_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.COIN_FOUND: 0.1,
    e.KILLED_OPPONENT: 5.5,
    e.GOT_KILLED: -2.5,
}

REWARD_PROFILES = {
    "standard": {
        "major": BASE_REWARDS,
        "shaping": {e.WAITED: -0.05, e.IN_DANGER: -0.2, e.REVERSED_DIRECTION: -0.015},
        "suicide": -2.0,
        "step_cost": 0.008,
        "clip": 0.1,
    },
    "coin": {
        "major": {**BASE_REWARDS, e.GOT_KILLED: -1.0},
        "shaping": {
            e.WAITED: -0.08,
            e.IN_DANGER: -0.02,
        },
        "suicide": -2.0,
        "step_cost": 0.008,
        "clip": 0.05,
    },
    "loot_exploration": {
        "major": {
            **BASE_REWARDS,
            e.GOT_KILLED: -1.0,
            e.COIN_FOUND: 0.5,
            e.CRATE_DESTROYED: 0.35,
            e.CRATE_BOMB_DROPPED: 0.05,
        },
        "shaping": {
            e.WAITED: -0.01,
            e.IN_DANGER: -0.04,
        },
        "suicide": -2.0,
        "step_cost": 0.0,
        "clip": 0.05,
    },
    "killer": {
        "major": {
            **BASE_REWARDS,
            e.COIN_COLLECTED: 0.15,
            e.COIN_FOUND: 0.0,
            e.KILLED_OPPONENT: 10.0,
            e.GOT_KILLED: -4.0,
            e.SURVIVED_ROUND: 0.75,
            e.BOMB_EXPLODED: 0.15,
            e.KILL_BOMB_DROPPED: 0.75,
        },
        "shaping": {
            e.WAITED: -0.02,
            e.IN_DANGER: -0.1,
        },
        "suicide": -6.0,
        "step_cost": 0.002,
        "clip": 0.15,
    },
}


PARAMETERS = (
    ("BUFFER_SIZE", "buffer_size", int),
    ("BATCH_SIZE", "batch_size", int),
    ("GAMMA", "gamma", float),
    ("LR", "lr", float),
    ("TARGET_UPDATE", "target_update", int),
    ("MIN_REPLAY_SIZE", "min_replay_size", int),
    ("TRAIN_EVERY_STEPS", "train_every_steps", int),
    ("END_OF_ROUND_OPT_STEPS", "end_of_round_opt_steps", int),
    ("EPSILON_START", "epsilon_start", float),
    ("EPSILON_END", "epsilon_end", float),
    ("EPSILON_HALF_LIFE", "epsilon_half_life", float),
    ("EVAL_EVERY_TRAINING_ROUNDS", "eval_every", int),
    ("EVAL_ROUNDS", "eval_rounds", int),
)

PROGRESS = (
    ("EPSILON_LAST", "epsilon_current", 0.8),
    ("STEPS_DONE", "steps_done", 0),
    ("BEST_MODEL_MEAN_GAME_SCORE", "best_mean_game_score", 0.0),
    ("BEST_MODEL_COINS_COLLECTED", "best_mean_coins_collected", 0.0),
    ("BEST_MODEL_MEAN_ENEMIES_KILLED", "best_mean_enemies_killed", 0.0),
    ("BEST_MODEL_COMPLETION_RATE", "best_completion_rate", 0.0),
    ("BEST_MODEL_MEAN_TIME_LEFT", "best_mean_time_left", 0.0),
    ("BEST_MODEL_SUICIDE_RATE", "best_suicide_rate", 1.0),
    ("BEST_MODEL_MEAN_CRATES_DESTROYED", "best_mean_crates_destroyed", 0.0),
    ("BEST_MODEL_SCORE", "best_score", -1.0e12),
)


def load_hyperparameters(self):
    values = {}
    with open(HYPERPARAMS_FILE, encoding="utf-8") as file:
        for line in file:
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = float(value.split("#", 1)[0].replace("_", ""))

    missing = [key for key, _, _ in PARAMETERS + PROGRESS if key not in values]
    if missing:
        raise ValueError("Missing hyperparameters: " + ", ".join(missing))

    for key, attribute, value_type in PARAMETERS:
        setattr(self, attribute, value_type(values[key]))
    for key, attribute, default in PROGRESS:
        value_type = int if attribute == "steps_done" else float
        setattr(self, attribute, value_type(values.get(key, default)))

    if self.batch_size > self.buffer_size or self.min_replay_size < self.batch_size:
        raise ValueError("Replay buffer must fit the warm-up and one batch")
    if min(self.train_every_steps, self.target_update, self.eval_every, self.eval_rounds) < 1:
        raise ValueError("Training and evaluation intervals must be positive")


def save_hyperparameters(self):
    """Rewrite the small text file, including every progress variable."""
    values = {}
    for key, attribute, _ in PARAMETERS + PROGRESS:
        values[key] = getattr(self, attribute)

    integer_keys = {
        "BUFFER_SIZE", "BATCH_SIZE", "TARGET_UPDATE", "MIN_REPLAY_SIZE",
        "TRAIN_EVERY_STEPS", "END_OF_ROUND_OPT_STEPS",
        "EVAL_EVERY_TRAINING_ROUNDS", "EVAL_ROUNDS", "STEPS_DONE",
    }
    with open(HYPERPARAMS_FILE, "w", encoding="utf-8") as file:
        for key, _, _ in PARAMETERS:
            value = int(values[key]) if key in integer_keys else values[key]
            file.write(f"{key}={value}\n")
        file.write("\n")
        for key, _, _ in PROGRESS:
            value = int(values[key]) if key in integer_keys else f"{values[key]:.10f}"
            file.write(f"{key}={value}\n")


def reset_progress(self):
    for _, attribute, default in PROGRESS:
        setattr(self, attribute, default)
    self.epsilon_current = self.epsilon_start


def save_checkpoint(self):
    torch.save(
        {
            "model": MODEL_NAME,
            "reward_profile": self.reward_profile_name,
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gradient_steps": self.gradient_steps,
            "training_rounds": self.training_rounds,
            "evaluation_round": self.evaluation_round,
            "evaluation_left": self.evaluation_left,
            "evaluation_results": self.evaluation_results,
        },
        LATEST_CHECKPOINT_FILE,
    )


def save_replay(self):
    with open(REPLAY_FILE, "wb") as file:
        pickle.dump(list(self.memory), file, protocol=pickle.HIGHEST_PROTOCOL)


def load_replay(self):
    if not os.path.isfile(REPLAY_FILE):
        return
    try:
        with open(REPLAY_FILE, "rb") as file:
            self.memory.extend(pickle.load(file)[-self.buffer_size :])
    except (OSError, pickle.PickleError, EOFError):
        self.logger.warning("Could not load replay buffer; starting with an empty one")


def setup_training(self):
    load_hyperparameters(self)
    self.reward_profile_name = os.environ.get("DQN_REWARD_PROFILE", "standard").lower()
    if self.reward_profile_name not in REWARD_PROFILES:
        raise ValueError("Unknown DQN_REWARD_PROFILE: " + self.reward_profile_name)
    self.reward_profile = REWARD_PROFILES[self.reward_profile_name]
    self.log_eval_events = os.environ.get("DQN_LOG_EVAL_EVENTS", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }

    self.memory = deque(maxlen=self.buffer_size)
    self.target_net = deepcopy(self.policy_net).to(self.device)
    self.target_net.eval()
    self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.lr)
    self.gradient_steps = 0
    self.training_rounds = 0
    self.evaluation_round = False
    self.evaluation_left = 0
    self.evaluation_results = []

    resume = getattr(self, "_resume_data", None)
    if self.training_mode == "resume" and resume:
        if resume.get("reward_profile") != self.reward_profile_name:
            self.logger.warning("Reward profile changed; keeping only the policy weights")
            self.training_mode = "transfer"
            reset_progress(self)
        else:
            self.target_net.load_state_dict(resume["target"])
            self.optimizer.load_state_dict(resume["optimizer"])
            self.gradient_steps = int(resume.get("gradient_steps", 0))
            self.training_rounds = int(resume.get("training_rounds", 0))
            self.evaluation_round = bool(resume.get("evaluation_round", False))
            self.evaluation_left = int(resume.get("evaluation_left", 0))
            self.evaluation_results = list(resume.get("evaluation_results", []))
            load_replay(self)
    elif self.training_mode != "resume":
        reset_progress(self)

    self.total_coins = int(os.environ.get("DQN_TOTAL_COINS", TOTAL_COINS))
    if self.total_coins < 1:
        raise ValueError("DQN_TOTAL_COINS must be positive")
    self.replay_saves_due = 0
    self._pending_key = None
    self._processed_key = None
    reset_round(self)
    save_hyperparameters(self)

    self.logger.info(
        "Training mode=%s profile=%s replay=%d epsilon=%.3f steps=%d eval_event_log=%s",
        self.training_mode, self.reward_profile_name, len(self.memory),
        self.epsilon_current, self.steps_done, self.log_eval_events,
    )


def pack(array):
    if array is None:
        return None
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def unpack(arrays):
    return np.stack(arrays).astype(np.float32) / 255.0


def advance_steps(self):
    self.steps_done += 1
    self.epsilon_current = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (
        0.5 ** (self.steps_done / self.epsilon_half_life)
    )


def optimize_model(self):
    population = list(self.memory)
    if self._pending_key is not None and population:
        population.pop()
    if len(population) < max(self.min_replay_size, self.batch_size):
        return None

    batch = random.sample(population, self.batch_size)
    boards = torch.from_numpy(unpack([item.board for item in batch])).to(self.device)
    scalars = torch.from_numpy(unpack([item.scalar for item in batch])).to(self.device)
    actions = torch.tensor([item.action for item in batch], device=self.device).unsqueeze(1)
    rewards = torch.tensor([item.reward for item in batch], device=self.device).unsqueeze(1)
    nonterminal = torch.tensor(
        [item.next_board is not None for item in batch], device=self.device
    ).unsqueeze(1)

    next_boards = torch.from_numpy(
        unpack([item.next_board if item.next_board is not None else item.board for item in batch])
    ).to(self.device)
    next_scalars = torch.from_numpy(
        unpack([item.next_scalar if item.next_scalar is not None else item.scalar for item in batch])
    ).to(self.device)
    next_masks = torch.from_numpy(
        np.stack([
            item.next_mask if item.next_mask is not None else np.ones(len(ACTIONS), dtype=np.uint8)
            for item in batch
        ]).astype(bool)
    ).to(self.device)

    current_q = self.policy_net(boards, scalars).gather(1, actions)
    with torch.no_grad():
        online_q = self.policy_net(next_boards, next_scalars).masked_fill(~next_masks, -torch.inf)
        next_actions = online_q.argmax(1, keepdim=True)
        next_q = self.target_net(next_boards, next_scalars).gather(1, next_actions)
        target_q = rewards + self.gamma * next_q * nonterminal

    loss = F.smooth_l1_loss(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
    self.optimizer.step()
    self.gradient_steps += 1

    if self.gradient_steps % self.target_update == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
    return float(loss.item())


def add_shaping_events(self, old_state, action, new_state, events):
    if old_state is None or new_state is None:
        return

    old_position = tuple(old_state["self"][-1])
    new_position = tuple(new_state["self"][-1])
    velocity = (new_position[0] - old_position[0], new_position[1] - old_position[1])
    if velocity != (0, 0) and getattr(self, "previous_velocity", (0, 0)) == (-velocity[0], -velocity[1]):
        events.append(e.REVERSED_DIRECTION)
    self.previous_velocity = velocity

    if new_position in {
        tuple(position) for position in np.argwhere(new_state.get("explosion_map", 0) > 0)
    }:
        events.append(e.IN_DANGER)

    if action == "BOMB" and e.BOMB_DROPPED in events:
        blast = blast_tiles(old_state["field"], old_position)
        enemies = {tuple(other[-1]) for other in old_state.get("others", [])}
        if enemies & blast:
            events.append(e.KILL_BOMB_DROPPED)
        elif any(old_state["field"][tile] == 1 for tile in blast):
            events.append(e.CRATE_BOMB_DROPPED)
        else:
            events.append(e.USELESS_BOMB_DROPPED)


def reward_from_events(self, events):
    major = self.reward_profile["major"]
    shaping = self.reward_profile["shaping"]
    suicided = e.KILLED_SELF in events
    major_reward = 0.0
    shaping_reward = 0.0

    for event in events:
        if event == e.KILLED_SELF or (suicided and event in (e.GOT_KILLED, e.BOMB_EXPLODED)):
            continue
        major_reward += major.get(event, 0.0)
        shaping_reward += shaping.get(event, 0.0)
    if suicided:
        major_reward += self.reward_profile["suicide"]

    limit = self.reward_profile["clip"]
    shaping_reward = float(np.clip(shaping_reward, -limit, limit))
    return major_reward + shaping_reward - self.reward_profile["step_cost"]


def transition_key(game_state, action):
    if game_state is None:
        return None
    return game_state.get("round"), game_state.get("step"), tuple(game_state["self"][-1]), action


def update_round_statistics(self, state, events):
    if state is not None:
        self.round_game_score = float(state["self"][1])
    self.round_coins += events.count(e.COIN_COLLECTED)
    self.round_kills += events.count(e.KILLED_OPPONENT)
    self.round_crates += events.count(e.CRATE_DESTROYED)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    events = list(events)
    self._processed_key = transition_key(old_game_state, self_action)
    update_round_statistics(self, new_game_state, events)
    add_shaping_events(self, old_game_state, self_action, new_game_state, events)
    if self.evaluation_round and getattr(self, "log_eval_events", False):
        self.round_event_counts.update(events)
    reward = reward_from_events(self, events)

    if (
        not self.round_completed
        and new_game_state is not None
        and self.round_coins >= self.total_coins
        and not new_game_state.get("others")
    ):
        self.round_completed = True
        self.round_time_left = max(0, s.MAX_STEPS - new_game_state.get("step", 0))
        reward += 5.0 + self.round_time_left / max(1, s.MAX_STEPS)

    if self.evaluation_round:
        return

    old_features = features_used_for_action(self, old_game_state)
    new_features = state_to_features(new_game_state, self_action)
    if old_features is None:
        return

    old_board, old_scalar = old_features
    if new_features is None:
        new_board = new_scalar = next_mask = None
    else:
        new_board, new_scalar = new_features
        next_mask = action_mask(new_game_state).astype(np.uint8)

    self.memory.append(
        Transition(
            pack(old_board), pack(old_scalar), ACTION_INDEX[self_action],
            pack(new_board), pack(new_scalar), next_mask, reward,
        )
    )
    self._pending_key = self._processed_key
    advance_steps(self)
    if self.steps_done % self.train_every_steps == 0:
        optimize_model(self)


def model_score(profile, mean_game_score, mean_kills, suicide_rate):
    if profile == "killer":
        return mean_game_score + 300.0 * mean_kills - 25.0 * suicide_rate
    return mean_game_score


def complete_evaluation(self):
    results = self.evaluation_results
    count = len(results)
    mean = lambda key: sum(result[key] for result in results) / count

    mean_game_score = mean("game_score")
    mean_coins = mean("coins")
    mean_kills = mean("kills")
    completion_rate = mean("completed")
    completed_times = [result["time_left"] for result in results if result["completed"]]
    mean_time_left = sum(completed_times) / len(completed_times) if completed_times else 0.0
    suicide_rate = mean("suicided")
    mean_crates = mean("crates")
    score = model_score(self.reward_profile_name, mean_game_score, mean_kills, suicide_rate)

    self.logger.info(
        "Evaluation: score=%.2f kills=%.2f coins=%.2f suicides=%.2f selection=%.2f",
        mean_game_score, mean_kills, mean_coins, suicide_rate, score,
    )
    if getattr(self, "log_eval_events", False):
        event_names = sorted({name for result in results for name in result.get("events", {})})
        event_means = [
            f"{name}={sum(result.get('events', {}).get(name, 0) for result in results) / count:.2f}"
            for name in event_names
        ]
        self.logger.info(
            "Evaluation mean events per round: %s",
            ", ".join(event_means) if event_means else "none",
        )
    if score > self.best_score:
        self.best_score = score
        self.best_mean_game_score = mean_game_score
        self.best_mean_coins_collected = mean_coins
        self.best_mean_enemies_killed = mean_kills
        self.best_completion_rate = completion_rate
        self.best_mean_time_left = mean_time_left
        self.best_suicide_rate = suicide_rate
        self.best_mean_crates_destroyed = mean_crates
        torch.save(self.policy_net.state_dict(), BEST_MODEL_FILE)
        self.logger.info("Saved new best model to %s", BEST_MODEL_FILE)
    self.evaluation_results = []


def end_of_round(self, last_game_state, last_action, events):
    events = list(events)
    key = transition_key(last_game_state, last_action)
    already_processed = key == self._processed_key

    if not already_processed:
        update_round_statistics(self, last_game_state, events)
        if last_game_state is not None:
            self.round_game_score += (
                events.count(e.COIN_COLLECTED) * s.REWARD_COIN
                + events.count(e.KILLED_OPPONENT) * s.REWARD_KILL
            )

    if self.evaluation_round:
        if getattr(self, "log_eval_events", False):
            if already_processed:
                self.round_event_counts.update(
                    event for event in events if event == e.SURVIVED_ROUND
                )
            else:
                self.round_event_counts.update(events)
        self.evaluation_results.append(
            {
                "game_score": self.round_game_score,
                "coins": self.round_coins,
                "kills": self.round_kills,
                "completed": float(self.round_completed),
                "time_left": self.round_time_left,
                "suicided": float(e.KILLED_SELF in events),
                "crates": self.round_crates,
                "events": dict(self.round_event_counts),
            }
        )
        self.evaluation_left -= 1
        if self.evaluation_left <= 0:
            complete_evaluation(self)
            self.evaluation_round = False
            self.training_rounds = 0
    else:
        if already_processed and self.memory:
            transition = self.memory[-1]
            extra_reward = self.reward_profile["major"].get(e.SURVIVED_ROUND, 0.0)
            self.memory[-1] = transition._replace(
                next_board=None, next_scalar=None, next_mask=None,
                reward=transition.reward + extra_reward,
            )
        else:
            features = features_used_for_action(self, last_game_state)
            if features is not None:
                board, scalar = features
                self.memory.append(
                    Transition(
                        pack(board), pack(scalar), ACTION_INDEX.get(last_action, ACTION_INDEX["WAIT"]),
                        None, None, None, reward_from_events(self, events),
                    )
                )
                advance_steps(self)
        self._pending_key = None

        for _ in range(self.end_of_round_opt_steps):
            optimize_model(self)
        self.training_rounds += 1
        if self.training_rounds >= self.eval_every:
            self.evaluation_round = True
            self.evaluation_left = self.eval_rounds
            self.evaluation_results = []
            self.logger.info("Starting %d greedy evaluation rounds", self.eval_rounds)

    save_checkpoint(self)
    save_hyperparameters(self)
    self.replay_saves_due += 1
    if self.replay_saves_due >= REPLAY_SAVE_EVERY_ROUNDS:
        save_replay(self)
        self.replay_saves_due = 0
    reset_round(self)


def reset_round(self):
    self.round_game_score = 0.0
    self.round_coins = 0
    self.round_kills = 0
    self.round_crates = 0
    self.round_completed = False
    self.round_time_left = 0
    self.round_event_counts = Counter()
    self.previous_velocity = (0, 0)
    self._processed_key = None
    self._pending_key = None
