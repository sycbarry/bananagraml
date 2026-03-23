
"""
This file is the environment for the BananaGraml game.
It is used to train the agent to play the game.
It is also used to test the agent's performance.
It is also used to play the game manually.


How it works: 

1. Reward signals (see `_compute_reward_delta` and module constants):

    - Small per-step cost to prefer shorter episodes.
    - Placing a tile from the bench onto the board (bench count drops, board count rises).
    - Bonus when a new orthogonal adjacency appears between board tiles (grid neighbors:
      up / down / left / right), including after repositioning on the grid.
    - Bonus when the number of valid dictionary word-lines on the grid increases; penalty when
      invalid word-lines increase (from `BananaGramlModel.analyze_board_words`, aligned with
      how `Game` / `Tile` commit moves via `place_tile_on_board` and validation).
    - Bonus / penalty when `model.board_valid` flips true/false after a `validate()` run.
    - Penalty when a newly isolated tile appears (no adjacent tiles on the logical grid).
    - Large bonus when `model.victory` becomes true (empty bench + valid board in game rules).
    - Tiny bonus when the agent starts holding a tile (`Game.selected_tile` becomes set).
    - On board focus, cursor-move actions (arrows) are penalized in proportion to how much the
      crosshair ends up farther from the nearest on-board tile (pixel distance).


2.  Game State (Observation Space): 

    - The tiles on the board. 
    - The tiles on the bench. 
    - The total valid words on the board. 
    - The total invalid words on the board. 
    - The position of the cursor on the board. 


"""





import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
import pygame.locals as locals

from game.main import Game, GameConfig, Tile
from game.src.game.model import BananaGramlModel

# --- Reward shaping (tune for your algorithm) ---
R_STEP_COST = -0.02
R_PLACE_FROM_BENCH = 0.75
# Per new orthogonal neighbor-pair on the logical grid (shared edge between two occupied cells).
R_ORTHOGONAL_ADJACENCY = 0.4
R_REPOSITION_WORD_CHANGE = 0.2
R_VALID_WORD_LINE = 2.5
R_INVALID_WORD_LINE = -1.25
R_BOARD_BECAME_VALID = 0.5
R_BOARD_BECAME_INVALID = -1.5
R_ISOLATED_TILE = -2.0
R_PICKUP_TILE = 0.05
R_VICTORY = 75.0
# When focus is BOARD and a cursor step increases distance to nearest board tile (pixels).
R_CURSOR_AWAY_FROM_NEAREST_BOARD_TILE = -0.02

board_dimensions = (
    # something like 1080*720
    # where 1080 is width and 720 is height
    GameConfig.BOARD_WIDTH,  # is longer than height
    GameConfig.BOARD_HEIGHT,  # is shorter than width.
    GameConfig.DIVIDER,
)

class BananaGramlEnvironment(gym.Env):

    def __init__(self):

        model = BananaGramlModel(board_dimensions)
        model.init_bench(10)
        self.game = Game(model)
        # Do not call game.run() here: it blocks until the window closes, so train.py
        # never reaches its loop (no prints / logs). The display is driven from render().

        # Scalar high must fit pixel coords (width can exceed BOARD_HEIGHT) and tile/word counts.
        _obs_hi = max(GameConfig.BOARD_WIDTH, GameConfig.BOARD_HEIGHT)
        self.observation_space = gym.spaces.Dict(
            {
                "board_valid": gym.spaces.Discrete(2),
                "board": gym.spaces.Box(0, _obs_hi, shape=(2,), dtype=np.int64),
                "bench": gym.spaces.Box(0, _obs_hi, shape=(2,), dtype=np.int64),
                "tiles_on_board": gym.spaces.Box(0, _obs_hi, shape=(2,), dtype=np.int64),
                "position_cursor": gym.spaces.Box(0, _obs_hi, shape=(2,), dtype=np.int64),
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

    @staticmethod
    def _orthogonal_adjacency_edges(cells: frozenset[Tuple[int, int]]) -> set[frozenset[Tuple[int, int]]]:
        """Undirected edges between orthogonally adjacent occupied cells."""
        edges: set[frozenset[Tuple[int, int]]] = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    edges.add(frozenset({(r, c), (nr, nc)}))
        return edges

    def _dist_cursor_to_nearest_board_tile(self) -> Optional[float]:
        """Pixel distance from board crosshair to closest `Tile` on the board, if any."""
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
        model = self.game.model
        words = model.analyze_board_words()
        return {
            "n_board": len(model.tiles_on_board),
            "n_bench": len(model.tiles_on_bench),
            "board_valid": bool(model.board_valid),
            "victory": bool(model.victory),
            "valid_words": int(words["valid_words"]),
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
        reward = R_STEP_COST

        d_valid = after["valid_words"] - before["valid_words"]
        d_invalid = after["invalid_words"] - before["invalid_words"]

        if d_valid > 0:
            reward += d_valid * R_VALID_WORD_LINE
            info["new_valid_word_lines"] = d_valid
        if d_invalid > 0:
            reward += d_invalid * R_INVALID_WORD_LINE
            info["new_invalid_word_lines"] = d_invalid

        # Tile left bench and appeared on board (matches keyboard/drag flows in `main.py`)
        placed_from_bench = (
            after["n_board"] > before["n_board"]
            and after["n_bench"] < before["n_bench"]
        )
        if placed_from_bench:
            reward += R_PLACE_FROM_BENCH
            info["placed_from_bench"] = True

        before_cells = before["grid_cells"]
        after_cells = after["grid_cells"]
        if before_cells != after_cells:
            eb = self._orthogonal_adjacency_edges(before_cells)
            ea = self._orthogonal_adjacency_edges(after_cells)
            new_edges = len(ea - eb)
            if new_edges > 0:
                reward += new_edges * R_ORTHOGONAL_ADJACENCY
                info["new_orthogonal_adjacency_edges"] = new_edges

        # Board topology changed without a bench->board transfer (e.g. moved on grid)
        if not placed_from_bench and (
            after["valid_words"] != before["valid_words"]
            or after["invalid_words"] != before["invalid_words"]
        ):
            reward += R_REPOSITION_WORD_CHANGE
            info["word_layout_changed_on_board"] = True

        if after["board_valid"] and not before["board_valid"]:
            reward += R_BOARD_BECAME_VALID
            info["model_board_became_valid"] = True
        if not after["board_valid"] and before["board_valid"]:
            reward += R_BOARD_BECAME_INVALID
            info["model_board_became_invalid"] = True

        if after["isolated_tile"] and not before["isolated_tile"]:
            reward += R_ISOLATED_TILE
            info["new_isolated_tile"] = True

        if after["holding"] and not before["holding"]:
            reward += R_PICKUP_TILE
            info["picked_up_tile"] = True

        # Penalize board crosshair moves that increase distance to the nearest board tile.
        cursor_actions = (0, 1, 2, 3)
        if int(action) in cursor_actions and before.get("focus_board"):
            db = before.get("dist_nearest_board_tile")
            da = after.get("dist_nearest_board_tile")
            if db is not None and da is not None and da > db:
                delta = da - db
                penalty = R_CURSOR_AWAY_FROM_NEAREST_BOARD_TILE * delta
                reward += penalty
                info["cursor_moved_away_from_nearest_board_px"] = delta
                info["cursor_away_penalty"] = penalty

        terminated = bool(after["victory"])
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
        # `get_game_state()` exposes ModelTile instances and a 2D grid — SB3 needs fixed numeric vectors.
        return {
            "board_valid": int(bool(m.board_valid)),
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
