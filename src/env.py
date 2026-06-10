import math
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pygame
import pygame.locals as locals

from game.main import Game, GameConfig
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
# Tile moved from hand to the board grid (count of board tiles increased).
R_TILE_TO_BOARD = 7.0
# Dealt new tiles after clearing the bench with a valid board.
R_PEEL = 12.0
# Bank empty / win condition from the model.
R_VICTORY = 150.0
# Small shaping when focus or pick-up state actually changes.
R_FOCUS_SWITCH = 0.1


######

board_dimensions = (
    GameConfig.BOARD_WIDTH,
    GameConfig.BOARD_HEIGHT,
    GameConfig.DIVIDER,
)


# Match BananaGramlModel.build_coordinates grid size (fixed shape for SB3 buffers).
_BOARD_COLS = math.floor(GameConfig.BOARD_WIDTH // GameConfig.DIVIDER)
_BOARD_ROWS = math.floor(GameConfig.BOARD_HEIGHT // GameConfig.DIVIDER)

# Fast A–Z / a–z → 1–26 for observation encoding (avoids branches per cell).
_LETTER_LUT = np.zeros(256, dtype=np.float32)
for _c in range(ord("A"), ord("Z") + 1):
    _LETTER_LUT[_c] = float(_c - ord("A") + 1)
for _c in range(ord("a"), ord("z") + 1):
    _LETTER_LUT[_c] = float(_c - ord("a") + 1)

_BOARD_VALID_OBS = (np.int64(0), np.int64(1))

_CURSOR_KEYS = (
    locals.K_UP,
    locals.K_DOWN,
    locals.K_LEFT,
    locals.K_RIGHT,
)


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

        # Reused each step to cut allocations (SB3 copies into its vec buffers).
        self._board_grid_buf = np.zeros((_BOARD_ROWS, _BOARD_COLS), dtype=np.float32)
        self._bench_buf = np.zeros((self._max_bench_tiles,), dtype=np.float32)
        self._cross_buf = np.zeros((2,), dtype=np.float32)

    def _compute_reward_delta(
        self, before: Dict[str, Any], after: Dict[str, Any], action: int
    ) -> tuple[float, bool, Dict[str, float]]:
        reward = 0.0
        terminated = False
        info: Dict[str, float] = {}

        n_board0 = int(before["_n_board"])
        n_board1 = int(after["_n_board"])
        n_bench0 = int(before["_n_bench"])
        n_bench1 = int(after["_n_bench"])
        dn_board = n_board1 - n_board0

        valid0 = bool(before["board_valid"])
        valid1 = bool(after["board_valid"])

        if dn_board > 0:
            reward += R_TILE_TO_BOARD
            info["r_tile_to_board"] = R_TILE_TO_BOARD

        # Bench was cleared with a valid layout, then new tiles were peeled.
        if n_bench0 == 0 and n_bench1 > 0 and valid1:
            reward += R_PEEL
            info["r_peel"] = R_PEEL

        if valid1 and not valid0:
            reward += R_BOARD_VALID
            info["r_recovered_valid"] = R_BOARD_VALID
        elif not valid1 and valid0:
            pen = R_BROKE_VALID_WORD * 2.0
            reward -= pen
            info["r_broke_valid"] = -pen

        if valid1:
            reward += 0.02
        else:
            reward -= 0.05

        if after["_victory"] and not before["_victory"]:
            reward += R_VICTORY
            terminated = True
            info["r_victory"] = R_VICTORY

        if action == 4 and before["_focus"] != after["_focus"]:
            reward += R_FOCUS_SWITCH
            info["r_focus_switch"] = R_FOCUS_SWITCH

        if action in (5, 6):
            if dn_board != 0 or before["_holding"] != after["_holding"]:
                ri = R_INTERACT * 0.5
                reward += ri
                info["r_interact"] = ri
            else:
                reward += 0.02
                info["r_interact_try"] = 0.02

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
        event = pygame.event.Event(
            locals.KEYDOWN, key=_CURSOR_KEYS[value], mod=locals.KMOD_NONE
        )
        pygame.event.post(event)

    def _reward_snapshot(self) -> Dict[str, Any]:
        m = self.model
        g = self.game
        snap = m.get_game_state()
        snap["_n_board"] = len(m.tiles_on_board)
        snap["_n_bench"] = len(m.tiles_on_bench)
        snap["_victory"] = bool(m.victory)
        snap["_focus"] = g.focus_area
        snap["_holding"] = g.selected_tile is not None
        return snap

    def _encode_board_grid(self) -> np.ndarray:
        grid = self._board_grid_buf
        grid.fill(0.0)
        board = self.model.board
        if (
            len(board) != _BOARD_ROWS
            or _BOARD_ROWS == 0
            or len(board[0]) != _BOARD_COLS
        ):
            return grid
        lut = _LETTER_LUT
        for i in range(_BOARD_ROWS):
            row = board[i]
            grow = grid[i]
            for j in range(_BOARD_COLS):
                cell = row[j]
                if cell is not None:
                    ch = cell.get_value()
                    if ch:
                        grow[j] = lut[ord(ch[0])]
        return grid

    def _encode_bench_letters(self) -> np.ndarray:
        buf = self._bench_buf
        buf.fill(0.0)
        bench = self.model.tiles_on_bench
        n = min(len(bench), self._max_bench_tiles)
        lut = _LETTER_LUT
        for k in range(n):
            ch = bench[k].get_value()
            if ch:
                buf[k] = lut[ord(ch[0])]
        return buf

    def _get_obs(self) -> Dict[str, Any]:
        pos = self.game.cross_hair_position
        self._cross_buf[0] = float(pos[0])
        self._cross_buf[1] = float(pos[1])
        return {
            "board_grid": self._encode_board_grid(),
            "bench_letters": self._encode_bench_letters(),
            "cross_hair_position": self._cross_buf,
            "board_valid": _BOARD_VALID_OBS[1 if self.model.board_valid else 0],
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
