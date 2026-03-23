
"""
This file is the environment for the BananaGraml game.
It is used to train the agent to play the game.
It is also used to test the agent's performance.
It is also used to play the game manually.


How it works: 

1. Reward signals (see `_compute_reward_delta` and module constants):

    - **No reward change** (0) when logical **board occupancy is unchanged** — same set of grid
      cells occupied as before the step (cursor moves, focus switch, pickup/drag that does not
      commit a different placement, etc. do not add or subtract score), **except** a small penalty
      when the bench is **empty** (all in-hand tiles are on the board) and focus still **switches
      to the bench** — discourages useless toggling.
    - When the board layout **does** change: word-based positives/negatives apply as below; neutral
      moves can still get **centering** shaping and the small **tile layout** bonus.
    - On editing steps: penalties when **invalid word-lines** increase, when the model reports
      **new isolated** tiles, when **`board_valid`** becomes false, and a large bonus on **victory**.
    - Extra shaping for **vertical (column) dictionary word-lines**: reward changes with
      `valid_column_words` from `analyze_board_words` (in addition to total `valid_words`).
    - **Cursor on board**: if focus stays on the **BOARD** and a cursor step **increases** distance
      to the **nearest** on-board tile (pixel space), apply a penalty — reduces idling in empty
      corners away from the tile cluster.
    - **Tile layout change**: small bonus whenever occupied grid cells change (reposition / place /
      pick up from board), on top of word-based rewards.
    - **Centering**: on layout-changing steps, small bonus when the **mean** tile distance to the
      logical grid center **decreases** (keeps the cluster off the edges).


2.  Game State (Observation Space): 

    - Dict includes: ``board_valid``, **``focus``** (bench vs board), word counts, bench/board sizes,
      cursor pixels, **``cursor_norm``** / **``dist_to_nearest_tile_norm``** in ``[0, 1]`` for stable learning.
    - **Letter state**: ``board_letters`` is a fixed (rows×cols) grid, 0 = empty, 1–26 = A–Z;
      ``rack_letters`` is a fixed-length padded vector for the bench rack.


"""





import math
import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
import pygame.locals as locals

from game.main import Game, GameConfig, Tile
from game.src.game.model import BananaGramlModel

# --- Reward shaping (tune for your algorithm) ---
# Applied only on steps where occupied grid cells change (see `_compute_reward_delta`).
# Per change in count of valid / invalid dictionary word-lines (`analyze_board_words`).
R_VALID_WORD_LINE = 2.5
# Per change in count of valid words read down a column (vertical lines only).
R_VALID_COLUMN_WORD_LINE = 1.25
R_INVALID_WORD_LINE = -1.25
R_BOARD_BECAME_INVALID = -1.5
R_ISOLATED_TILE = -2.0
R_VICTORY = 75.0
# Bench has no tiles (`n_bench == 0`) but focus switches from board → bench (no placement step).
R_FOCUS_BENCH_WHEN_BENCH_EMPTY = -1
# Board focus, cursor move: penalty per pixel the crosshair ends farther from the nearest tile.
R_CURSOR_AWAY_FROM_NEAREST_TILE = -0.02
# Any step that changes which logical grid cells are occupied (tile moved / placed / lifted).
R_TILE_LAYOUT_CHANGE = 0.06
# Mean grid (row/col) Euclidean distance to board center decreases (grid units, ~0–20).
R_TILES_CLOSER_TO_CENTER = 0.12

board_dimensions = (
    # something like 1080*720
    # where 1080 is width and 720 is height
    GameConfig.BOARD_WIDTH,  # is longer than height
    GameConfig.BOARD_HEIGHT,  # is shorter than width.
    GameConfig.DIVIDER,
)

# Fewer tiles = easier early learning (try 5–7). Full Bananagrams feel: 10+.
_INIT_BENCH_TILES = int(os.environ.get("BANANAGRAML_INIT_BENCH", "10"))
# Padded bench / rack observation (letters beyond this are truncated for the policy).
_MAX_RACK_LETTERS = 32


def _letter_to_idx(value: Optional[str]) -> int:
    """0 = empty / unknown; 1–26 = A–Z."""
    if not value:
        return 0
    s = str(value).strip().upper()
    if not s:
        return 0
    ch = s[0]
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 1
    return 0


class BananaGramlEnvironment(gym.Env):

    def __init__(self):

        model = BananaGramlModel(board_dimensions)
        model.init_bench(_INIT_BENCH_TILES)
        self.game = Game(model)
        # Do not call game.run() here: it blocks until the window closes, so train.py
        # never reaches its loop (no prints / logs). The display is driven from render().

        coords = self.game.model.coordinates
        self._grid_rows = len(coords)
        self._grid_cols = len(coords[0]) if self._grid_rows else 0

        # Scalar high must fit pixel coords (width can exceed BOARD_HEIGHT) and tile/word counts.
        _obs_hi = max(GameConfig.BOARD_WIDTH, GameConfig.BOARD_HEIGHT)
        self.observation_space = gym.spaces.Dict(
            {
                "board_valid": gym.spaces.Discrete(2),
                # 0 = bench focus, 1 = board focus (policy must know where arrows apply).
                "focus": gym.spaces.Discrete(2),
                "board": gym.spaces.Box(0, _obs_hi, shape=(2,), dtype=np.int64),
                "bench": gym.spaces.Box(0, _obs_hi, shape=(2,), dtype=np.int64),
                "tiles_on_board": gym.spaces.Box(0, _obs_hi, shape=(2,), dtype=np.int64),
                "position_cursor": gym.spaces.Box(0, _obs_hi, shape=(2,), dtype=np.int64),
                # Normalized features (stable for PPO + VecNormalize).
                "cursor_norm": gym.spaces.Box(
                    0.0, 1.0, shape=(2,), dtype=np.float32
                ),
                "dist_to_nearest_tile_norm": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                # 0 empty, 1–26 letter (fixed logical grid from model.coordinates).
                "board_letters": gym.spaces.Box(
                    0,
                    27,
                    (self._grid_rows, self._grid_cols),
                    dtype=np.int64,
                ),
                "rack_letters": gym.spaces.Box(
                    0,
                    27,
                    (_MAX_RACK_LETTERS,),
                    dtype=np.int64,
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(7)

        print(self.observation_space, flush=True)

        self.total_rewards = 0.0
        self._display_alive = True

        self.focus_area = "BOARD"

    def _board_grid_cells(self) -> frozenset[Tuple[int, int]]:
        """Logical grid indices (row, col) of tiles on the board, aligned with `coordinate_ref`."""
        model = self.game.model
        cells: set[Tuple[int, int]] = set()
        for t in model.tiles_on_board:
            if not isinstance(t, Tile):
                continue
            center = t.model_tile.get_position()
            if center in model.coordinate_ref:
                cells.add(model.coordinate_ref[center])
        return frozenset(cells)

    def _mean_dist_tiles_to_grid_center(
        self, cells: frozenset[Tuple[int, int]]
    ) -> Optional[float]:
        """Mean Euclidean distance in grid indices from occupied cells to the board center."""
        if not cells:
            return None
        coords = self.game.model.coordinates
        rows, cols = len(coords), len(coords[0])
        cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
        total = 0.0
        for r, c in cells:
            total += math.hypot(r - cr, c - cc)
        return total / len(cells)

    def _dist_cursor_to_nearest_board_tile(self) -> Optional[float]:
        """Pixel distance from board crosshair to closest on-board tile (stalling in corners increases this)."""
        cx, cy = self.game.cross_hair_position
        best: Optional[float] = None
        for t in self.game.model.tiles_on_board:
            if isinstance(t, Tile):
                px, py = t.rect.center
                d = math.hypot(cx - px, cy - py)
                if best is None or d < best:
                    best = d
        return best

    def _board_letter_grid(self) -> np.ndarray:
        """Letter index per logical cell; 0 = empty."""
        g = np.zeros((self._grid_rows, self._grid_cols), dtype=np.int64)
        m = self.game.model
        for t in m.tiles_on_board:
            if not isinstance(t, Tile):
                continue
            mt = t.model_tile
            center = mt.get_position()
            if center not in m.coordinate_ref:
                continue
            r, c = m.coordinate_ref[center]
            g[r, c] = _letter_to_idx(mt.get_value())
        return g

    def _rack_letter_vector(self) -> np.ndarray:
        """Bench tiles left-to-right in model order, padded with zeros."""
        v = np.zeros((_MAX_RACK_LETTERS,), dtype=np.int64)
        bench = self.game.model.tiles_on_bench
        for i, mt in enumerate(bench[:_MAX_RACK_LETTERS]):
            v[i] = _letter_to_idx(mt.get_value())
        return v

    def _analyze_words_for_rewards(self) -> Dict[str, Any]:
        """
        Re-scan the logical grid for word lines vs the dictionary.
        Rewards that mention valid/invalid words or isolation must use this (via `_reward_snapshot`),
        not stale counts — call only after the step’s game events have been applied.
        """
        return self.game.model.analyze_board_words()

    def _reward_snapshot(self) -> Dict[str, Any]:
        model = self.game.model
        words = self._analyze_words_for_rewards()
        return {
            "n_board": len(model.tiles_on_board),
            "n_bench": len(model.tiles_on_bench),
            "board_valid": bool(model.board_valid),
            "victory": bool(model.victory),
            "valid_words": int(words["valid_words"]),
            "valid_column_words": int(words.get("valid_column_words", 0)),
            "invalid_words": int(words["invalid_words"]),
            "isolated_tile": bool(words["isolated_tile"]),
            "holding": self.game.selected_tile is not None,
            "bank_remaining": model.tile_bank.get_current_size(),
            "focus_board": self.game.focus_area == "BOARD",
            "dist_nearest_board_tile": self._dist_cursor_to_nearest_board_tile(),
            "grid_cells": self._board_grid_cells(),
        }

    def _compute_reward_delta(
        self, before: Dict[str, Any], after: Dict[str, Any], action: int
    ) -> Tuple[float, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {"action": int(action)}
        terminated = bool(after["victory"])
        board_layout_changed = before["grid_cells"] != after["grid_cells"]

        if not board_layout_changed:
            reward = 0.0
            if (
                after["n_bench"] == 0
                and after["n_board"] > 0
                and before["focus_board"]
                and not after["focus_board"]
            ):
                reward += R_FOCUS_BENCH_WHEN_BENCH_EMPTY
                info["focus_bench_while_bench_empty"] = True
            cursor_actions = (0, 1, 2, 3)
            if (
                int(action) in cursor_actions
                and before.get("focus_board")
                and after.get("focus_board")
                and before.get("dist_nearest_board_tile") is not None
            ):
                db = before["dist_nearest_board_tile"]
                da = after.get("dist_nearest_board_tile")
                if da is not None and da > db:
                    delta = da - db
                    penalty = R_CURSOR_AWAY_FROM_NEAREST_TILE * delta
                    reward += penalty
                    info["cursor_moved_away_from_nearest_board_tile_px"] = delta
                    info["cursor_away_penalty"] = penalty
            if terminated:
                reward += R_VICTORY
                info["victory"] = True
            info["board_layout_unchanged"] = True
            info["reward_total"] = reward
            return reward, terminated, info

        reward = 0.0
        reward += R_TILE_LAYOUT_CHANGE
        info["tile_layout_changed"] = True

        d_valid = after["valid_words"] - before["valid_words"]
        d_invalid = after["invalid_words"] - before["invalid_words"]

        if d_valid != 0:
            reward += d_valid * R_VALID_WORD_LINE
            info["delta_valid_word_lines"] = d_valid
        d_valid_col = after["valid_column_words"] - before["valid_column_words"]
        if d_valid_col != 0:
            reward += d_valid_col * R_VALID_COLUMN_WORD_LINE
            info["delta_valid_column_word_lines"] = d_valid_col
        if d_invalid > 0:
            reward += d_invalid * R_INVALID_WORD_LINE
            info["new_invalid_word_lines"] = d_invalid

        if not after["board_valid"] and before["board_valid"]:
            reward += R_BOARD_BECAME_INVALID
            info["model_board_became_invalid"] = True

        if after["isolated_tile"] and not before["isolated_tile"]:
            reward += R_ISOLATED_TILE
            info["new_isolated_tile"] = True

        db = self._mean_dist_tiles_to_grid_center(before["grid_cells"])
        da = self._mean_dist_tiles_to_grid_center(after["grid_cells"])
        if db is not None and da is not None:
            closer = db - da
            if closer != 0.0:
                reward += R_TILES_CLOSER_TO_CENTER * closer
                info["delta_mean_dist_to_board_center"] = closer

        if terminated:
            reward += R_VICTORY
            info["victory"] = True

        info["reward_total"] = reward
        return reward, terminated, info

    def step(self, action):
        action = int(action)

        before = self._reward_snapshot()

        if action == 0:
            self.move_cursor(0)
        elif action == 1:
            self.move_cursor(1)
        elif action == 2:
            self.move_cursor(2)
        elif action == 3:
            self.move_cursor(3)
        elif action == 4:
            self.switch_focus_area()
        elif action == 5:
            self.pickup_tile_from_position()
        elif action == 6:
            self.drop_tile_at_position()

        if self._display_alive:
            running = self.game.handle_events()
            if not running:
                pygame.quit()
                self._display_alive = False

        after = self._reward_snapshot()
        reward, terminated, info = self._compute_reward_delta(before, after, action)
        self.total_rewards += reward

        obs = self._get_obs()
        truncated = False
        return obs, reward, terminated, truncated, info

    def pickup_tile_from_position(self): 
        event = pygame.event.Event(locals.KEYDOWN, key=locals.K_x, mod=locals.KMOD_NONE)
        pygame.event.post(event)

    def drop_tile_at_position(self): 
        event = pygame.event.Event(locals.KEYDOWN, key=locals.K_x, mod=locals.KMOD_NONE)
        pygame.event.post(event)

    def switch_focus_area(self): 
        event = pygame.event.Event(locals.KEYDOWN, key=locals.K_z, mod=locals.KMOD_NONE)
        self.focus_area = "BOARD" if self.focus_area == "BENCH" else "BENCH"
        pygame.event.post(event)

    def move_cursor(self, value): 
        directions = {
            0: locals.K_UP, 
            1: locals.K_DOWN, 
            2: locals.K_LEFT, 
            3: locals.K_RIGHT
        }
        event = pygame.event.Event(locals.KEYDOWN, key=directions[value], mod=locals.KMOD_NONE)
        pygame.event.post(event)


    def _get_obs(self):
        g, m = self.game, self.game.model
        words = m.analyze_board_words()
        cx, cy = g.cross_hair_position
        d = self._dist_cursor_to_nearest_board_tile()
        diag = math.hypot(float(GameConfig.BOARD_WIDTH), float(GameConfig.BOARD_HEIGHT))
        dist_norm = 1.0 if d is None else float(d) / diag
        cursor_norm = np.asarray(
            [cx / float(GameConfig.BOARD_WIDTH), cy / float(GameConfig.BOARD_HEIGHT)],
            dtype=np.float32,
        )
        return {
            "board_valid": int(bool(m.board_valid)),
            "focus": 1 if g.focus_area == "BOARD" else 0,
            "board": np.asarray(
                [words["valid_words"], words["invalid_words"]], dtype=np.int64
            ),
            "bench": np.asarray(
                [len(m.tiles_on_bench), m.tile_bank.get_current_size()], dtype=np.int64
            ),
            "tiles_on_board": np.asarray(
                [len(m.tiles_on_board), int(words["isolated_tile"])], dtype=np.int64
            ),
            "position_cursor": np.asarray([int(cx), int(cy)], dtype=np.int64),
            "cursor_norm": cursor_norm,
            "dist_to_nearest_tile_norm": np.asarray([dist_norm], dtype=np.float32),
            "board_letters": self._board_letter_grid(),
            "rack_letters": self._rack_letter_vector(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_rewards = 0.0
        self.focus_area = self.game.focus_area
        return self._get_obs(), {}

    def render(self):
        if self._display_alive:
            self._display_alive = self.game.tick_frame()

    def close(self): 
        self.game.kill()
