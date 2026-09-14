import os
import pickle
import random
from collections import deque, namedtuple

import numpy as np

import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Wie lange ein neu gewähltes Gegner-Angriffsziel mindestens beibehalten wird,
# solange es erreichbar und nicht gefährlich ist.
OPPONENT_TARGET_LOCK_STEPS = 3

DIRECTIONS = [((0, -1), 0), ((1, 0), 1), ((0, 1), 2), ((-1, 0), 3)]

Q_TABLE_FILE = os.path.join(
    os.path.dirname(__file__),
    "q_table.pkl"
)

StateFeatures = namedtuple("StateFeatures", [
    "target_direction",
    "free_up",
    "free_right",
    "free_down",
    "free_left",
    "bomb_available",
    "crate_in_blast_range",
    "in_danger",
    "danger_urgency",
    "coin_visible",
    "safe_bomb",
    "opponent_in_blast_range",
    "movement_pattern",
    "loop_return_direction",
])


def valid_actions_from_state(state):

    valid_actions = []

    if state.free_up:
        valid_actions.append("UP")

    if state.free_right:
        valid_actions.append("RIGHT")

    if state.free_down:
        valid_actions.append("DOWN")

    if state.free_left:
        valid_actions.append("LEFT")


    valid_actions.append("WAIT")

    if state.bomb_available:
        valid_actions.append("BOMB")

    return valid_actions


def setup(self):
    self.epsilon = 0.1

    if os.path.isfile(Q_TABLE_FILE):
        self.logger.info("Loading Q-table.")
        try:
            with open(Q_TABLE_FILE, "rb") as file:
                self.q_table = pickle.load(file)
        except (TypeError, AttributeError, ValueError, pickle.UnpicklingError) as exc:
            # Feature-Layout hat sich zwischen Agent-Versionen geaendert.
            # Eine inkompatible Tabelle darf nicht still weiterverwendet werden.
            self.logger.warning(
                "Could not load Q-table with current StateFeatures (%s). "
                "Starting with an empty table.",
                exc,
            )
            self.q_table = {}
    else:
        self.logger.info("Starting with empty Q-table.")
        self.q_table = {}

    self.current_target_position = None
    self.current_target_kind = None
    self.opponent_target_lock_until = None

    self.last_state = None
    self.last_target_position = None
    self.last_distance = None
    self.position_history = deque(maxlen=6)


def detect_movement_pattern(position_history, new_position=None):
    """
    Return:
        movement_pattern,
        loop_return_direction

    movement_pattern:
        0 = normal
        1 = Zwei-Feld-/Rueckkehr-Loop (A -> B -> A)
        2 = Stagnation (A -> A -> A)

    loop_return_direction:
        0-3 = Richtung vom aktuellen Feld zum unmittelbar
              vorherigen Feld
        4   = keine relevante Rueckkehr-Richtung
    """

    positions = list(position_history)

    if new_position is not None:
        positions.append(tuple(new_position))

    if len(positions) >= 3:
        current = positions[-1]
        previous = positions[-2]
        before_previous = positions[-3]

        # -----------------------------------------
        # A -> A -> A
        # Agent bewegt sich ueberhaupt nicht.
        # -----------------------------------------
        if current == previous == before_previous:
            return 2, 4

        # -----------------------------------------
        # A -> B -> A
        # Klassischer Zwei-Feld-Loop.
        # -----------------------------------------
        if current == before_previous and current != previous:

            dx = previous[0] - current[0]
            dy = previous[1] - current[1]

            for (move_dx, move_dy), direction_id in DIRECTIONS:

                if (dx, dy) == (move_dx, move_dy):
                    return 1, direction_id

    # -----------------------------------------
    # Falls der Zwei-Feld-Loop bereits laenger
    # laeuft, weiterhin erkennen.
    # -----------------------------------------
    if len(positions) >= 6:

        recent = positions[-6:]

        if (
            len(set(recent)) <= 2
            and recent[-1] != recent[-2]
        ):

            current = recent[-1]
            previous = recent[-2]

            dx = previous[0] - current[0]
            dy = previous[1] - current[1]

            for (move_dx, move_dy), direction_id in DIRECTIONS:

                if (dx, dy) == (move_dx, move_dy):
                    return 1, direction_id

    return 0, 4


def act(self, game_state: dict) -> str:

    current_position = tuple(game_state["self"][3])
    self.position_history.append(current_position)
    movement_pattern, loop_return_direction = detect_movement_pattern(
        self.position_history
    )

    update_current_target(self, game_state)

    state = state_to_features(
        game_state,
        self.current_target_position,
        movement_pattern=movement_pattern,
        loop_return_direction=loop_return_direction,
    )

    self.last_state = state
    self.last_target_position = self.current_target_position
    self.last_distance = distance_to_target(
        game_state,
        self.current_target_position
    )

    valid_actions = valid_actions_from_state(state)

    if self.train and random.random() < self.epsilon:

        # Noch nicht ausprobierte Aktionen bevorzugen.
        # Dadurch werden besonders seltene Danger- und Loop-States
        # schneller vollstaendig gelernt.
        unseen_actions = [
            action
            for action in valid_actions
            if (state, action) not in self.q_table
        ]

        if unseen_actions:
            return random.choice(unseen_actions)

        return random.choice(valid_actions)

    # WICHTIG: Ein nie gelernter Q-Wert darf bei der Evaluation nicht
    # automatisch als Q=0 eine bereits gelernte negative Aktion schlagen.
    # Im Training bleibt epsilon-greedy fuer echte Exploration erhalten.
    known_q_values = {
        action: self.q_table[(state, action)]
        for action in valid_actions
        if (state, action) in self.q_table
    }

    if known_q_values:
        max_q = max(known_q_values.values())
        best_actions = [
            action
            for action, value in known_q_values.items()
            if value == max_q
        ]
        return random.choice(best_actions)

    # Vollstaendig unbekannter Zustand: keine Rule-Policy erzwingen,
    # sondern nur unter den physisch gueltigen Aktionen waehlen.
    return random.choice(valid_actions)

def state_to_features(game_state: dict,target_position=None,movement_pattern=0,loop_return_direction=4,):

    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bomb_available, (x, y) = game_state["self"]

    danger_tiles = compute_danger_tiles(game_state)
    danger_urgency = get_danger_urgency(game_state, (x, y))
    in_danger = danger_urgency > 0

    free_up = is_free(game_state, x, y - 1)
    free_right = is_free(game_state, x + 1, y)
    free_down = is_free(game_state, x, y + 1)
    free_left = is_free(game_state, x - 1, y)

    crate_in_blast_range = has_crate_in_blast_range(field, (x, y))
    coin_visible = len(game_state["coins"]) > 0

    opponent_in_blast_range = can_hit_opponent_from_position(
        game_state,
        (x, y)
    )

    safe_bomb = False
    if bomb_available and not in_danger:
        safe_bomb = can_escape_own_bomb(game_state, danger_tiles)

    target_direction, _ = get_target_direction_and_distance(
        game_state, danger_tiles, in_danger, target_position,
    )

    return StateFeatures(
        target_direction,
        free_up,
        free_right,
        free_down,
        free_left,
        bool(bomb_available),
        crate_in_blast_range,
        in_danger,
        danger_urgency,
        coin_visible,
        safe_bomb,
        opponent_in_blast_range,
        movement_pattern,
        loop_return_direction,
    )

def distance_to_target(game_state, target_position=None):

    if game_state is None:
        return None

    _, _, _, (x, y) = game_state["self"]
    danger_tiles = compute_danger_tiles(game_state)
    in_danger = (x, y) in danger_tiles

    _, distance = get_target_direction_and_distance(
        game_state, danger_tiles, in_danger, target_position,
    )
    return distance


def get_target_direction_and_distance(
    game_state,
    danger_tiles,
    in_danger,
    target_position=None
):


    if in_danger:
        return direction_to_safety(game_state, danger_tiles)

    if target_position is not None:
        return direction_to_position(game_state, target_position)


    if game_state["coins"]:
        return direction_to_coin(game_state)

    return direction_to_crate(game_state)


def is_free(game_state, x, y):


    field = game_state["field"]

    if x < 0 or y < 0:
        return False

    if x >= field.shape[0] or y >= field.shape[1]:
        return False

    if field[x, y] != 0:
        return False

    bomb_positions = {
        position
        for position, _timer in game_state["bombs"]
    }

    if (x, y) in bomb_positions:
        return False

    other_positions = {
        position
        for _name, _score, _bombs_left, position in game_state["others"]
    }

    if (x, y) in other_positions:
        return False

    return True


def has_adjacent_crate(field, position):
    x, y = position

    for (dx, dy), _direction_id in DIRECTIONS:
        nx, ny = x + dx, y + dy

        if (
            0 <= nx < field.shape[0]
            and 0 <= ny < field.shape[1]
            and field[nx, ny] == 1
        ):
            return True

    return False


def has_crate_in_blast_range(field, position):
    """Return whether a bomb at *position* could currently hit a crate.

    Das Feature bildet die tatsaechliche Bomben-Utility besser ab als nur
    "Kiste direkt benachbart".
    """
    blast_tiles = get_blast_coords(
        field,
        position[0],
        position[1],
        s.BOMB_POWER,
    )

    return any(
        field[x, y] == 1
        for x, y in blast_tiles
        if (x, y) != tuple(position)
    )



def can_hit_opponent_from_position(game_state, position):
    if game_state is None or not game_state["others"]:
        return False

    field = game_state["field"]

    opponent_positions = {
        tuple(other[3])
        for other in game_state["others"]
    }

    blast_tiles = set(
        get_blast_coords(
            field,
            position[0],
            position[1],
            s.BOMB_POWER
        )
    )

    return bool(opponent_positions & blast_tiles)



def bfs_target_position(game_state, is_goal, block_danger=True):

    _, _, _, start = game_state["self"]

    danger_tiles = (
        compute_danger_tiles(game_state)
        if block_danger
        else set()
    )

    if is_goal(start):
        return start

    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()

        for (dx, dy), _direction_id in DIRECTIONS:
            nx, ny = x + dx, y + dy
            next_position = (nx, ny)

            if next_position in visited:
                continue

            if not is_free(game_state, nx, ny):
                continue

            if next_position in danger_tiles:
                continue

            visited.add(next_position)

            if is_goal(next_position):
                return next_position

            queue.append(next_position)

    return None


def update_current_target(self, game_state):

    if game_state is None:
        return

    field = game_state["field"]
    _, _, bomb_available, position = game_state["self"]

    danger_tiles = compute_danger_tiles(game_state)

    def is_valid_crate_target(pos):
        if not has_adjacent_crate(field, pos):
            return False

        if pos in danger_tiles:
            return False

        if not bomb_available:
            return True

        return can_escape_bomb_from_position(
            game_state,
            pos,
            danger_tiles
        )

    def is_valid_opponent_target(pos):

        if pos in danger_tiles:
            return False

        if not can_hit_opponent_from_position(game_state, pos):
            return False

        if bomb_available:
            return can_escape_bomb_from_position(
                game_state,
                pos,
                danger_tiles
            )

        return True

    # Wenn wir bereits akut in Gefahr sind,
    # wird die normale Zielwahl ausgesetzt.
    if position in danger_tiles:
        return

    coins = set(game_state["coins"])

    # -------------------------------------------------------
    # Agent_4_1:
    # 1. Direkte sichere Angriffsmöglichkeit hat Priorität.
    # -------------------------------------------------------
    if (
        game_state["others"]
        and bomb_available
        and can_hit_opponent_from_position(game_state, position)
        and can_escape_bomb_from_position(
            game_state,
            position,
            danger_tiles
        )
    ):
        self.current_target_position = position
        self.current_target_kind = "opponent"
        self.opponent_target_lock_until = None
        return

    # -------------------------------------------------------
    # 2. Nahe Münzen zuerst einsammeln.
    # -------------------------------------------------------
    if coins:
        _, coin_distance = direction_to_coin(game_state)

        if coin_distance is not None and coin_distance <= 5:

            if (
                self.current_target_kind == "coin"
                and self.current_target_position in coins
            ):
                _, current_coin_distance = direction_to_position(
                    game_state,
                    self.current_target_position
                )

                if (
                    current_coin_distance is not None
                    and current_coin_distance <= 5
                ):
                    return

            new_target = bfs_target_position(
                game_state,
                lambda pos: pos in coins,
                block_danger=True,
            )

            if new_target is not None:
                self.current_target_position = new_target
                self.current_target_kind = "coin"
                self.opponent_target_lock_until = None
                return

    # -------------------------------------------------------
    # 3. Wenn keine nahe Muenze existiert:
    #    Gegner aktiv verfolgen.
    #
    # Ein bereits gewaehltes Angriffsziel wird kurz stabil gehalten.
    # So springt die Zielposition nicht bei jeder gegnerischen Bewegung
    # von links nach rechts und erzeugt dadurch selbst einen Bewegungsloop.
    # -------------------------------------------------------
    if game_state["others"]:

        if (
            self.current_target_kind == "opponent"
            and self.current_target_position is not None
            and self.current_target_position not in danger_tiles
        ):
            _, target_distance = direction_to_position(
                game_state,
                self.current_target_position,
            )

            if target_distance is not None:
                if is_valid_opponent_target(self.current_target_position):
                    self.opponent_target_lock_until = (
                        game_state["step"] + OPPONENT_TARGET_LOCK_STEPS
                    )
                    return

                if (
                    target_distance > 0
                    and self.opponent_target_lock_until is not None
                    and game_state["step"] <= self.opponent_target_lock_until
                ):
                    return

        self.opponent_target_lock_until = None

        new_target = bfs_target_position(
            game_state,
            is_valid_opponent_target,
            block_danger=True,
        )

        if new_target is not None:
            self.current_target_position = new_target
            self.current_target_kind = "opponent"
            self.opponent_target_lock_until = (
                game_state["step"] + OPPONENT_TARGET_LOCK_STEPS
            )
            return


    if coins:

        if (
            self.current_target_kind == "coin"
            and self.current_target_position in coins
        ):
            return

        new_target = bfs_target_position(
            game_state,
            lambda pos: pos in coins,
            block_danger=True,
        )

        self.current_target_position = new_target
        self.current_target_kind = (
            "coin" if new_target is not None else None
        )
        self.opponent_target_lock_until = None
        return

    if (
        self.current_target_kind == "crate"
        and self.current_target_position is not None
        and is_valid_crate_target(self.current_target_position)
    ):
        return

    new_target = bfs_target_position(
        game_state,
        is_valid_crate_target,
        block_danger=True,
    )

    self.current_target_position = new_target
    self.current_target_kind = (
        "crate" if new_target is not None else None
    )
    self.opponent_target_lock_until = None


def bfs_target(game_state, is_goal, block_danger, danger_tiles=None):

    _, _, _, start = game_state["self"]

    if danger_tiles is None:
        danger_tiles = compute_danger_tiles(game_state) if block_danger else set()
    elif not block_danger:
        danger_tiles = set()

    if is_goal(start):
        return 4, 0

    queue = deque()
    visited = {start}

    for (dx, dy), direction_id in DIRECTIONS:
        nx, ny = start[0] + dx, start[1] + dy

        if is_free(game_state, nx, ny) and (nx, ny) not in danger_tiles:
            visited.add((nx, ny))
            queue.append(((nx, ny), direction_id, 1))

    while queue:
        position, first_direction, distance = queue.popleft()

        if is_goal(position):
            return first_direction, distance

        x, y = position

        for (dx, dy), _direction_id in DIRECTIONS:
            nx, ny = x + dx, y + dy
            next_position = (nx, ny)

            if (
                next_position not in visited
                and is_free(game_state, nx, ny)
                and next_position not in danger_tiles
            ):
                visited.add(next_position)
                queue.append((next_position, first_direction, distance + 1))

    return 5, None


def direction_to_coin(game_state):
    coins = set(game_state["coins"])
    return bfs_target(game_state, lambda pos: pos in coins, block_danger=True)


def direction_to_crate(game_state):
    field = game_state["field"]
    return bfs_target(
        game_state,
        lambda pos: has_adjacent_crate(field, pos),
        block_danger=True,
    )


def direction_to_safety(game_state, danger_tiles=None):
    if danger_tiles is None:
        danger_tiles = compute_danger_tiles(game_state)

    return bfs_target(
        game_state,
        lambda pos: pos not in danger_tiles,
        block_danger=False,
    )


def direction_to_position(game_state, target_position):
    if target_position is None:
        return 4, None

    return bfs_target(
        game_state,
        lambda pos: pos == target_position,
        block_danger=True,
    )


def get_blast_coords(field, x, y, power):

    blast_coords = [(x, y)]

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i

            if field[nx, ny] == -1:
                break

            blast_coords.append((nx, ny))

    return blast_coords


def get_danger_urgency(game_state, position):
    """Encode how urgently *position* must be left.

    0 = currently no explosion path reaches the tile
    1 = a bomb can reach it, but not in the next immediate phase
    2 = active explosion or a bomb with timer <= 1 can reach it

    Der Timer wird bewusst grob gebucketed, damit die Q-Tabelle nicht durch
    vier einzelne Timerwerte unnoetig aufgeblasen wird.
    """
    if game_state is None:
        return 0

    x, y = position

    if game_state["explosion_map"][x, y] > 0:
        return 2

    field = game_state["field"]
    urgency = 0

    for (bx, by), timer in game_state["bombs"]:
        blast_tiles = get_blast_coords(
            field,
            bx,
            by,
            s.BOMB_POWER,
        )

        if position not in blast_tiles:
            continue

        if timer <= 1:
            return 2

        urgency = 1

    return urgency


def compute_danger_tiles(game_state):

    field = game_state["field"]
    danger_tiles = set()

    explosion_map = game_state["explosion_map"]
    xs, ys = np.nonzero(explosion_map > 0)
    danger_tiles.update(zip(xs.tolist(), ys.tolist()))

    for (bx, by), _timer in game_state["bombs"]:
        danger_tiles.update(get_blast_coords(field, bx, by, s.BOMB_POWER))

    return danger_tiles


def can_escape_bomb_from_position(game_state, start, danger_tiles=None):

    field = game_state["field"]

    if danger_tiles is None:
        danger_tiles = compute_danger_tiles(game_state)

    own_blast = set(
        get_blast_coords(
            field,
            start[0],
            start[1],
            s.BOMB_POWER
        )
    )

    combined_danger = danger_tiles | own_blast

    max_steps = max(s.BOMB_TIMER - 1, 1)

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (px, py), distance = queue.popleft()

        if (px, py) not in combined_danger:
            return True

        if distance >= max_steps:
            continue

        for (dx, dy), _direction_id in DIRECTIONS:
            nx, ny = px + dx, py + dy
            next_position = (nx, ny)

            if (
                next_position not in visited
                and is_free(game_state, nx, ny)
            ):
                visited.add(next_position)
                queue.append((next_position, distance + 1))

    return False

def can_escape_own_bomb(game_state, danger_tiles=None):
    _, _, bomb_available, position = game_state["self"]

    if not bomb_available:
        return False

    return can_escape_bomb_from_position(
        game_state,
        position,
        danger_tiles
    )
