import os
import random
from collections import deque

import numpy as np
import torch

import settings as s
from .Networks import DQN


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
BOARD_CHANNELS = 7
SCALAR_SIZE = 4 + len(ACTIONS)
TOTAL_COINS = int(
    os.environ.get("DQN_TOTAL_COINS", s.SCENARIOS["classic"]["COIN_COUNT"])
)

DIRECTORY = os.path.dirname(__file__)
BEST_MODEL_FILE = os.path.join(DIRECTORY, "best-model.pt")
LATEST_CHECKPOINT_FILE = os.path.join(DIRECTORY, "latest-checkpoint.pt")
PRETRAINED_MODEL_FILE = os.path.join(DIRECTORY, "pretrained-model.pt")
TRAINING_MODES = {"fresh", "resume", "transfer"}


def blast_tiles(field, position):
    """Tiles hit by a bomb at position."""
    tiles = {tuple(position)}
    x, y = position
    for dx, dy in MOVES[:4]:
        for distance in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * distance, y + dy * distance
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:
                break
            tiles.add((nx, ny))
    return tiles


def bomb_is_useful(game_state):
    field = game_state["field"]
    position = tuple(game_state["self"][-1])
    blast = blast_tiles(field, position)
    enemies = {tuple(other[-1]) for other in game_state.get("others", [])}
    return bool(enemies & blast) or any(field[x, y] == 1 for x, y in blast)


def bomb_has_escape(game_state):
    """Simple static search for a tile outside our prospective blast."""
    field = game_state["field"]
    start = tuple(game_state["self"][-1])
    blast = blast_tiles(field, start)
    blocked = {tuple(position) for position, _ in game_state.get("bombs", [])}
    blocked |= {tuple(other[-1]) for other in game_state.get("others", [])}
    queue = deque([(start, 0)])
    seen = {start}

    while queue:
        position, distance = queue.popleft()
        if position not in blast:
            return True
        if distance >= s.BOMB_TIMER:
            continue
        for dx, dy in MOVES:
            nxt = (position[0] + dx, position[1] + dy)
            x, y = nxt
            if nxt in seen or nxt in blocked:
                continue
            if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 0:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return False


def immediate_danger(game_state):
    field = game_state["field"]
    danger = {
        tuple(position)
        for position in np.argwhere(game_state.get("explosion_map", np.zeros_like(field)) > 0)
    }
    for position, timer in game_state.get("bombs", []):
        if timer <= 1:
            danger |= blast_tiles(field, position)
    return danger


def action_mask(game_state):
    """Remove illegal actions and obvious one-step suicides."""
    if game_state is None:
        return np.zeros(len(ACTIONS), dtype=bool)

    field = game_state["field"]
    x, y = game_state["self"][-1]
    occupied = {tuple(position) for position, _ in game_state.get("bombs", [])}
    occupied |= {tuple(other[-1]) for other in game_state.get("others", [])}
    danger = immediate_danger(game_state)
    legal = []

    for dx, dy in MOVES:
        nxt = (x + dx, y + dy)
        nx, ny = nxt
        legal.append(
            0 <= nx < field.shape[0]
            and 0 <= ny < field.shape[1]
            and field[nx, ny] == 0
            and nxt not in occupied
        )

    can_bomb = (
        bool(game_state["self"][2])
        and (x, y) not in occupied
        and bomb_is_useful(game_state)
        and bomb_has_escape(game_state)
    )
    legal.append(can_bomb)
    legal = np.asarray(legal, dtype=bool)

    safe = legal.copy()
    for index, (dx, dy) in enumerate(MOVES):
        safe[index] = safe[index] and (x + dx, y + dy) not in danger
    result = safe if safe.any() else legal
    if not result.any():
        result[ACTIONS.index("WAIT")] = True
    return result


def state_to_features(game_state, last_action=None):
    """Turn a game state into seven full-board maps and ten numbers."""
    if game_state is None:
        return None

    field = game_state["field"]
    width, height = field.shape
    coins = game_state.get("coins", [])
    bombs = game_state.get("bombs", [])
    others = game_state.get("others", [])
    explosion_map = game_state.get("explosion_map", np.zeros_like(field))

    coin_map = np.zeros((width, height), dtype=np.float32)
    self_map = np.zeros_like(coin_map)
    enemy_map = np.zeros_like(coin_map)
    bomb_map = np.zeros_like(coin_map)
    danger_map = (explosion_map > 0).astype(np.float32)

    for position in coins:
        coin_map[tuple(position)] = 1
    self_map[tuple(game_state["self"][-1])] = 1
    for other in others:
        enemy_map[tuple(other[-1])] = 1
    for position, timer in bombs:
        bomb_map[tuple(position)] = (timer + 1) / (s.BOMB_TIMER + 1)
        strength = (s.BOMB_TIMER - timer + 1) / (s.BOMB_TIMER + 1)
        for tile in blast_tiles(field, position):
            danger_map[tile] = max(danger_map[tile], strength)

    board = np.stack(
        (field == -1, field == 1, coin_map, self_map, enemy_map, bomb_map, danger_map)
    ).astype(np.float32)

    last_action = "WAIT" if last_action not in ACTIONS else last_action
    action_one_hot = np.zeros(len(ACTIONS), dtype=np.float32)
    action_one_hot[ACTIONS.index(last_action)] = 1
    scalar = np.concatenate(
        (
            np.array(
                [
                    float(game_state["self"][2]),
                    len(coins) / max(1, TOTAL_COINS),
                    len(others) / max(1, s.MAX_AGENTS - 1),
                    game_state.get("step", 0) / max(1, s.MAX_STEPS),
                ],
                dtype=np.float32,
            ),
            action_one_hot,
        )
    )
    return board, scalar


def _state_key(game_state):
    if game_state is None:
        return None
    return game_state.get("round"), game_state.get("step"), tuple(game_state["self"][-1])


def features_used_for_action(self, game_state):
    if getattr(self, "_acted_key", None) == _state_key(game_state):
        return self._acted_features
    return state_to_features(game_state, getattr(self, "last_action", "WAIT"))


def setup(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.policy_net = DQN(
        BOARD_CHANNELS, (s.COLS, s.ROWS), SCALAR_SIZE, len(ACTIONS)
    ).to(self.device)
    self.last_action = "WAIT"
    self._acted_key = None
    self._acted_features = None
    self._resume_data = None

    if self.train:
        self.training_mode = os.environ.get("DQN_TRAINING_MODE", "resume").lower()
        if self.training_mode not in TRAINING_MODES:
            raise ValueError("DQN_TRAINING_MODE must be fresh, resume, or transfer")
        if self.training_mode == "resume":
            model_path = LATEST_CHECKPOINT_FILE
        elif self.training_mode == "transfer":
            model_path = os.environ.get("DQN_PRETRAINED_MODEL", PRETRAINED_MODEL_FILE)
        else:
            model_path = None
    else:
        self.training_mode = None
        model_path = BEST_MODEL_FILE if os.path.isfile(BEST_MODEL_FILE) else LATEST_CHECKPOINT_FILE

    if model_path and os.path.isfile(model_path):
        data = torch.load(model_path, map_location=self.device, weights_only=True)
        weights = data.get("policy", data) if isinstance(data, dict) else data
        try:
            self.policy_net.load_state_dict(weights)
            if self.train and self.training_mode == "resume" and "policy" in data:
                self._resume_data = data
            self.logger.info("Loaded model from %s", model_path)
        except RuntimeError:
            if self.training_mode == "transfer":
                raise
            self.logger.warning("Ignored incompatible checkpoint %s", model_path)
            if self.train:
                self.training_mode = "fresh"
    elif self.train and self.training_mode == "transfer":
        raise FileNotFoundError(model_path)
    elif self.train and self.training_mode == "resume":
        self.training_mode = "fresh"

    self.policy_net.eval()


def act(self, game_state):
    features = state_to_features(game_state, self.last_action)
    if features is None:
        return "WAIT"

    self._acted_key = _state_key(game_state)
    self._acted_features = features
    allowed = action_mask(game_state)
    choices = [action for action, valid in zip(ACTIONS, allowed) if valid]
    if not choices:
        choices = ["WAIT"]

    epsilon = 0.0
    if self.train and not getattr(self, "evaluation_round", False):
        epsilon = getattr(self, "epsilon_current", 1.0)

    if random.random() < epsilon:
        action = random.choice(choices)
    else:
        board, scalar = features
        with torch.no_grad():
            q_values = self.policy_net(
                torch.from_numpy(board).unsqueeze(0).to(self.device),
                torch.from_numpy(scalar).unsqueeze(0).to(self.device),
            )[0]
            q_values[~torch.from_numpy(allowed).to(self.device)] = -torch.inf
            action = ACTIONS[int(q_values.argmax().item())]

    self.last_action = action
    return action
