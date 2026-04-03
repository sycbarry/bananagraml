
import math
import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
import pygame.locals as locals

from game.main import Game, GameConfig, Tile
from game.src.game.model import BananaGramlModel



# Scores ######


# Pressing interact (pick up / place / confirm).
R_INTERACT = 3.0
# New dictionary-valid horizontal word line.
R_NEW_ROW_WORD = 10.0
# New dictionary-valid vertical word line.
R_NEW_COL_WORD = 10.0
# Extra bonus on top of row/column counts when any new valid word line appears.
R_NEW_WORD_LINK = 22.0
# Penalty per lost valid word line (multiplier; d_valid is negative).
R_BROKE_VALID_WORD = 4.0
# Centroid moved closer to board center (scale on normalized distance drop).
R_CENTER_TOWARD = 2.5

R_BOARD_VALID = 10


######

board_dimensions = (
    GameConfig.BOARD_WIDTH,
    GameConfig.BOARD_HEIGHT,
    GameConfig.DIVIDER,
)


# Match BananaGramlModel.build_coordinates grid size (fixed shape for SB3 buffers).
_BOARD_COLS = math.floor(GameConfig.BOARD_WIDTH // GameConfig.DIVIDER)
_BOARD_ROWS = math.floor(GameConfig.BOARD_HEIGHT // GameConfig.DIVIDER)


def _letter_slot(value: str) -> float:
    if not value or not str(value).isalpha():
        return 0.0
    return float(ord(str(value).upper()[0]) - ord("A") + 1)


class BananaGramlEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        *,
        starting_tiles_on_bench: int = 10,
        max_bench_tiles: int = 32,
    ):
        if max_bench_tiles < 1:
            raise ValueError("max_bench_tiles must be at least 1")
        if starting_tiles_on_bench < 0:
            raise ValueError("starting_tiles_on_bench must be non-negative")

        self._max_bench_tiles = max_bench_tiles

        self.model = BananaGramlModel(board_dimensions)
        self.model.init_bench(starting_tiles_on_bench)
        self.game = Game(self.model)

        """
        What are out observations:
        These are the things the agent can immediately perceive
        1. The locations of tiles on the board
        2. How many tiles are on the bench
        3. Valid words on the board
        4. The crosshair position
        """
        self.observation_space = gym.spaces.Dict({
            "board_grid": gym.spaces.Box(
                low=0.0,
                high=26.0,
                shape=(_BOARD_ROWS, _BOARD_COLS),
                dtype=np.float32,
            ),
            "bench_letters": gym.spaces.Box(
                low=0.0,
                high=26.0,
                shape=(self._max_bench_tiles,),
                dtype=np.float32,
            ),
            "cross_hair_position": gym.spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array(
                    [float(GameConfig.SCREEN_WIDTH), float(GameConfig.SCREEN_HEIGHT)],
                    dtype=np.float32,
                ),
                shape=(2,),
                dtype=np.float32,
            ),
            "board_valid": gym.spaces.Discrete(2),
        })

        """
        These are all the possible actions the agent can take in 
        the game. We should include all possible actions regardless of 
        whether they are meaningful or not. 
        1. Move right
        2. Move up
        3. Move down 
        4. Move left 
        5. Interact with a tile
        7. Switch from bench to board
        """
        self.action_space = gym.spaces.Discrete(7)

        self.render_mode = render_mode

        self.total_rewards = 0.0
        self._display_alive = True

    def _compute_reward_delta(self, before: Dict[str, Any], after: Dict[str, Any], action: int):
        reward = 0 
        terminated = False 
        info = {}
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
        elif action in (5, 6):
            # Both indices post K_x (pick up, drop, or bench→board); kept as two IDs for
            # compatibility with policies trained with Discrete(7).
            self._press_interact_key()

        if self._display_alive:
            running = self.game.handle_events()
            if not running:
                pygame.quit()
                self._display_alive = False
            elif self.render_mode == "human":
                # handle_events() does not draw; without this the window stays black.
                self.game.render()

        after = self._reward_snapshot()
        reward, terminated, info = self._compute_reward_delta(before, after, action)
        self.total_rewards += reward

        obs = self._get_obs()
        truncated = False
        return obs, reward, terminated, truncated, info

    def _press_interact_key(self) -> None:
        pygame.event.post(pygame.event.Event(locals.KEYDOWN, key=locals.K_x, mod=locals.KMOD_NONE))

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

    def _reward_snapshot(self) -> Dict[str, Any]:
        return self.model.get_game_state()

    def _encode_board_grid(self) -> np.ndarray:
        grid = np.zeros((_BOARD_ROWS, _BOARD_COLS), dtype=np.float32)
        board = self.model.board
        if (
            len(board) != _BOARD_ROWS
            or _BOARD_ROWS == 0
            or len(board[0]) != _BOARD_COLS
        ):
            return grid
        for i in range(_BOARD_ROWS):
            for j in range(_BOARD_COLS):
                cell = board[i][j]
                if cell is not None:
                    grid[i, j] = _letter_slot(cell.get_value())
        return grid

    def _encode_bench_letters(self) -> np.ndarray:
        out = np.zeros((self._max_bench_tiles,), dtype=np.float32)
        bench = self.model.tiles_on_bench
        for k, tile in enumerate(bench[: self._max_bench_tiles]):
            out[k] = _letter_slot(tile.get_value())
        return out

    def _get_obs(self) -> Dict[str, Any]:
        pos = self.game.cross_hair_position
        cross = np.asarray(pos, dtype=np.float32).reshape(2)
        return {
            "board_grid": self._encode_board_grid(),
            "bench_letters": self._encode_bench_letters(),
            "cross_hair_position": cross,
            "board_valid": np.int64(1 if self.model.board_valid else 0),
        }



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_rewards = 0.0
        obs = self._get_obs()
        if self._display_alive and self.render_mode == "human":
            self.game.render()
        return obs, {}

    def render(self):
        if self._display_alive and self.render_mode == "human":
            self.game.render()

    def close(self):
        self.game.kill()
