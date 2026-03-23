
"""
BananaGraml RL environment.

**Rewards** (Bananagrams-oriented; applied **every** env step from post-step state):

    - **Large** bonus when the board has **at least one dictionary-valid word line**
      (`analyze_board_words` / ``valid_words > 0``).
    - **Medium** bonus when **any** tiles share an edge horizontally or vertically (connected cluster).
    - **Extra** for **new** valid word lines vs. the previous step; **penalty** for new invalid lines.
    - **Small** penalty when the grid reports an **isolated** tile (not connected).
    - **Bench**: negative reward while focus is **BENCH** and the rack is **empty**.

**Observations**: ``focus``, ``cursor_norm``, ``dist_to_nearest_tile_norm``, ``board_letters``,
``bench_letters``, ``game_stats``.

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

# --- Rewards (tune for PPO + VecNormalize clip_reward) ---
# At least one dictionary-valid word line on the board (big signal).
R_VALID_WORDS_PRESENT = 12.0
# Any orthogonal neighbor pairs among occupied cells (horizontal or vertical touch).
R_TILES_CONNECTED = 3.0
# Each *new* valid word line in one step (on top of R_VALID_WORDS_PRESENT when applicable).
R_NEW_VALID_WORD_LINE = 4.0
# Each *new* invalid word line.
R_NEW_INVALID_WORD_LINE = -2.0
# Isolated tile on the logical grid (disconnected from cluster).
R_ISOLATED_TILE = -1.0
# Focus on bench while rack is empty.
R_FOCUS_BENCH_WHEN_EMPTY = -0.35

board_dimensions = (
    GameConfig.BOARD_WIDTH,
    GameConfig.BOARD_HEIGHT,
    GameConfig.DIVIDER,
)

_INIT_BENCH_TILES = int(os.environ.get("BANANAGRAML_INIT_BENCH", "10"))
_MAX_BENCH_LETTERS = 32
_TILE_POOL_SIZE = 144


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

        coords = self.game.model.coordinates
        self._grid_rows = len(coords)
        self._grid_cols = len(coords[0]) if self._grid_rows else 0
        self._max_grid_cells = max(1, self._grid_rows * self._grid_cols)

        self.observation_space = gym.spaces.Dict(
            {
                "focus": gym.spaces.Discrete(2),
                "cursor_norm": gym.spaces.Box(
                    0.0, 1.0, shape=(2,), dtype=np.float32
                ),
                "board_letters": gym.spaces.Box(
                    0,
                    27,
                    (self._grid_rows, self._grid_cols),
                    dtype=np.int64,
                ),
                "bench_letters": gym.spaces.Box(
                    0,
                    27,
                    (_MAX_BENCH_LETTERS,),
                    dtype=np.int64,
                ),
                "dist_to_nearest_tile_norm": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "game_stats": gym.spaces.Box(
                    0.0, 1.0, shape=(8,), dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(7)

        print(self.observation_space, flush=True)

        self.total_rewards = 0.0
        self._display_alive = True

    def _board_grid_cells(self) -> frozenset[Tuple[int, int]]:
        model = self.game.model
        cells: set[Tuple[int, int]] = set()
        for t in model.tiles_on_board:
            if not isinstance(t, Tile):
                continue
            center = t.model_tile.get_position()
            if center in model.coordinate_ref:
                cells.add(model.coordinate_ref[center])
        return frozenset(cells)

    @staticmethod
    def _row_col_neighbor_pair_counts(
        cells: frozenset[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """Horizontal and vertical adjacency edge counts (each edge once)."""
        if not cells:
            return 0, 0
        horizontal = 0
        vertical = 0
        for r, c in cells:
            if (r, c + 1) in cells:
                horizontal += 1
            if (r + 1, c) in cells:
                vertical += 1
        return horizontal, vertical

    def _word_scan(self) -> Dict[str, Any]:
        return self.game.model.analyze_board_words()

    def _dist_cursor_to_nearest_board_tile(self) -> Optional[float]:
        cx, cy = self.game.cross_hair_position
        best: Optional[float] = None
        for t in self.game.model.tiles_on_board:
            if isinstance(t, Tile):
                px, py = t.rect.center
                d = math.hypot(cx - px, cy - py)
                if best is None or d < best:
                    best = d
        return best

    def _reward_snapshot(self) -> Dict[str, Any]:
        m = self.game.model
        w = self._word_scan()
        return {
            "victory": bool(m.victory),
            "grid_cells": self._board_grid_cells(),
            "n_bench": len(m.tiles_on_bench),
            "focus_board": self.game.focus_area == "BOARD",
            "valid_words": int(w["valid_words"]),
            "invalid_words": int(w["invalid_words"]),
            "isolated_tile": bool(w["isolated_tile"]),
        }

    def _apply_focus_bench_empty_penalty(
        self, after: Dict[str, Any], reward: float, info: Dict[str, Any]
    ) -> float:
        if after["n_bench"] == 0 and not after["focus_board"]:
            reward += R_FOCUS_BENCH_WHEN_EMPTY
            info["focus_bench_while_empty"] = True
        return reward

    def _compute_reward_delta(
        self, before: Dict[str, Any], after: Dict[str, Any], action: int
    ) -> Tuple[float, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {"action": int(action)}
        terminated = bool(after["victory"])
        reward = 0.0

        if after["valid_words"] > 0:
            reward += R_VALID_WORDS_PRESENT
            info["has_valid_dictionary_words"] = True

        h, v = self._row_col_neighbor_pair_counts(after["grid_cells"])
        pairs = h + v
        if pairs > 0:
            reward += R_TILES_CONNECTED
            info["orthogonal_neighbor_pairs"] = pairs

        d_valid = after["valid_words"] - before["valid_words"]
        if d_valid > 0:
            reward += d_valid * R_NEW_VALID_WORD_LINE
            info["new_valid_word_lines"] = d_valid

        d_invalid = after["invalid_words"] - before["invalid_words"]
        if d_invalid > 0:
            reward += d_invalid * R_NEW_INVALID_WORD_LINE
            info["new_invalid_word_lines"] = d_invalid

        if after["isolated_tile"] and not before["isolated_tile"]:
            reward += R_ISOLATED_TILE
            info["new_isolated_tile"] = True

        reward = self._apply_focus_bench_empty_penalty(after, reward, info)

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
        pygame.event.post(event)

    def move_cursor(self, value):
        directions = {
            0: locals.K_UP,
            1: locals.K_DOWN,
            2: locals.K_LEFT,
            3: locals.K_RIGHT,
        }
        event = pygame.event.Event(
            locals.KEYDOWN, key=directions[value], mod=locals.KMOD_NONE
        )
        pygame.event.post(event)

    def _board_letter_grid(self) -> np.ndarray:
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

    def _bench_letter_vector(self) -> np.ndarray:
        v = np.zeros((_MAX_BENCH_LETTERS,), dtype=np.int64)
        bench = self.game.model.tiles_on_bench
        for i, mt in enumerate(bench[:_MAX_BENCH_LETTERS]):
            v[i] = _letter_to_idx(mt.get_value())
        return v

    def _game_stats_vector(self) -> np.ndarray:
        m, g = self.game.model, self.game
        v = np.zeros(8, dtype=np.float32)
        n_bench = len(m.tiles_on_bench)
        n_board = len(m.tiles_on_board)
        v[0] = min(1.0, n_bench / float(_MAX_BENCH_LETTERS))
        v[1] = min(1.0, n_board / float(self._max_grid_cells))
        v[2] = min(1.0, m.tile_bank.get_current_size() / float(_TILE_POOL_SIZE))
        v[3] = 1.0 if g.selected_tile is not None else 0.0
        v[4] = 1.0 if m.board_valid else 0.0
        v[5] = 1.0 if m.tile_bank.can_dump() else 0.0
        cells = self._board_grid_cells()
        if cells:
            mr = sum(r for r, _ in cells) / len(cells)
            mc = sum(c for _, c in cells) / len(cells)
            rn = max(1.0, float(self._grid_rows - 1))
            cn = max(1.0, float(self._grid_cols - 1))
            v[6] = float(mr / rn)
            v[7] = float(mc / cn)
        else:
            v[6] = v[7] = 0.5
        return v

    def _get_obs(self):
        g = self.game
        cx, cy = g.cross_hair_position
        cursor_norm = np.asarray(
            [cx / float(GameConfig.BOARD_WIDTH), cy / float(GameConfig.BOARD_HEIGHT)],
            dtype=np.float32,
        )
        d = self._dist_cursor_to_nearest_board_tile()
        diag = math.hypot(float(GameConfig.BOARD_WIDTH), float(GameConfig.BOARD_HEIGHT))
        dist_norm = 1.0 if d is None else float(min(1.0, d / diag))
        return {
            "focus": 1 if g.focus_area == "BOARD" else 0,
            "cursor_norm": cursor_norm,
            "board_letters": self._board_letter_grid(),
            "bench_letters": self._bench_letter_vector(),
            "dist_to_nearest_tile_norm": np.asarray([dist_norm], dtype=np.float32),
            "game_stats": self._game_stats_vector(),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_rewards = 0.0
        return self._get_obs(), {}

    def render(self):
        if self._display_alive:
            self._display_alive = self.game.tick_frame()

    def close(self):
        self.game.kill()
