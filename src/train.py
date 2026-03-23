"""
Bananagrams-style RL training tips (env vars):

  BANANAGRAML_HEADLESS=1     Skip pygame each step (much faster; use for serious runs).
  BANANAGRAML_INIT_BENCH=5   Start with fewer rack tiles so valid words happen sooner (curriculum).
  BANANAGRAML_MAX_EPISODE_STEPS=8000  Truncate long stalls.

After training, load both the PPO checkpoint and `bananagraml_vec_normalize.pkl` for inference
(VecNormalize was used to scale obs/reward).
"""

import os

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import BananaGramlEnvironment

TOTAL_TIMESTEPS = 40_000
LOG_EVERY_N_ENV_STEPS = 10
HEADLESS = os.environ.get("BANANAGRAML_HEADLESS", "0") == "1"
MAX_EPISODE_STEPS = int(os.environ.get("BANANAGRAML_MAX_EPISODE_STEPS", "8000"))

# VecNormalize only normalizes Box spaces; Discrete keys stay as-is (board_valid, focus).
# board_letters / rack_letters are 0–27 letter indices — leave unnormalized (do not blur categories).
_VEC_NORM_OBS_KEYS = (
    "board",
    "bench",
    "tiles_on_board",
    "position_cursor",
    "cursor_norm",
    "dist_to_nearest_tile_norm",
)


def _base_gym_env(vec_env):
    """Unwrap VecNormalize → VecEnv → TimeLimit → BananaGramlEnvironment."""
    inner = getattr(vec_env, "venv", vec_env)
    envs = getattr(inner, "envs", None)
    if envs:
        return envs[0].unwrapped
    return vec_env.unwrapped


class LiveBoardCallback(BaseCallback):
    """
    PPO.learn() never calls render(); the old manual loop did every step.
    This runs pygame (handle_events + draw) once per collected env step.
    """

    def _on_training_start(self) -> None:
        _base_gym_env(self.training_env).render()

    def _on_step(self) -> bool:
        _base_gym_env(self.training_env).render()
        return True


class TrainingProgressCallback(BaseCallback):
    """Print cumulative env reward and a snapshot of game state every N env steps."""

    def __init__(self, every_n_steps: int = LOG_EVERY_N_ENV_STEPS):
        super().__init__()
        self.every_n_steps = every_n_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps != 0:
            return True
        env = _base_gym_env(self.training_env)
        g, m = env.game, env.game.model
        print(
            f"[env_step={self.n_calls} num_timesteps={self.num_timesteps}] "
            f"total_reward={env.total_rewards:.3f} | "
            f"bench_tiles={len(m.tiles_on_bench)} board_tiles={len(m.tiles_on_board)} "
            f"bank_remaining={m.tile_bank.get_current_size()} | "
            f"board_valid={m.board_valid} victory={m.victory} focus={g.focus_area}",
            flush=True,
        )
        return True


def _make_vec_env():
    def _thunk():
        return TimeLimit(
            BananaGramlEnvironment(),
            max_episode_steps=MAX_EPISODE_STEPS,
        )

    return _thunk


def main():
    venv = DummyVecEnv([_make_vec_env()])
    # Stabilizes mixed int/float Dict obs and return scale; clip_reward must fit R_VICTORY etc.
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=100.0,
        norm_obs_keys=list(_VEC_NORM_OBS_KEYS),
    )
    model = PPO(
        "MultiInputPolicy",
        venv,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.06,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs",
    )
    callbacks = [TrainingProgressCallback(LOG_EVERY_N_ENV_STEPS)]
    if not HEADLESS:
        callbacks.insert(0, LiveBoardCallback())
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList(callbacks),
    )
    model.save("bananagraml_ppo")
    venv.save("bananagraml_vec_normalize.pkl")
    venv.close()


if __name__ == "__main__":
    main()
