"""
Bananagrams-style RL training tips (env vars):

  BANANAGRAML_HEADLESS=1     Skip pygame each step (much faster; use for serious runs).
  BANANAGRAML_INIT_BENCH=5   Start with fewer rack tiles so valid words happen sooner (curriculum).
  BANANAGRAML_MAX_EPISODE_STEPS=8000  Truncate long stalls.

CLI examples:

  python train.py --run-name exp1 --timesteps 90000 --seed 42
  tensorboard --logdir ./tensorboard_logs

Parallel runs (see train_parallel.sh): separate --run-name per process, one TensorBoard logdir.

After training, load both the PPO checkpoint and matching `*_vec_normalize.pkl` for inference.
"""

import argparse
import importlib.util
import os
import sys

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import BananaGramlEnvironment

LOG_EVERY_N_ENV_STEPS = 10
HEADLESS = os.environ.get("BANANAGRAML_HEADLESS", "0") == "1"
MAX_EPISODE_STEPS = int(os.environ.get("BANANAGRAML_MAX_EPISODE_STEPS", "8000"))

# VecNormalize: only normalize continuous features; focus is Discrete; letters are categorical 0–27.
_VEC_NORM_OBS_KEYS = ("cursor_norm", "dist_to_nearest_tile_norm", "game_stats")


def _base_gym_env(vec_env):
    """Unwrap VecNormalize → VecEnv → TimeLimit → BananaGramlEnvironment."""
    inner = getattr(vec_env, "venv", vec_env)
    envs = getattr(inner, "envs", None)
    if envs:
        return envs[0].unwrapped
    return vec_env.unwrapped


def _tensorboard_log_dir(base_dir: str, run_name: str) -> str | None:
    if os.environ.get("BANANAGRAML_TENSORBOARD", "1") == "0":
        return None
    if importlib.util.find_spec("tensorboard") is None:
        return None
    path = os.path.join(base_dir, run_name)
    os.makedirs(path, exist_ok=True)
    return path


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


class TensorboardGameCallback(BaseCallback):
    """Extra TensorBoard scalars (SB3 already logs rollout/ from Monitor; this adds game state)."""

    def __init__(self, log_every: int = 32):
        super().__init__()
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.logger is None or self.n_calls % self.log_every != 0:
            return True
        env = _base_gym_env(self.training_env)
        m = env.game.model
        self.logger.record("game/total_reward", float(env.total_rewards))
        self.logger.record("game/bench_tiles", float(len(m.tiles_on_bench)))
        self.logger.record("game/board_tiles", float(len(m.tiles_on_board)))
        self.logger.record("game/bank_remaining", float(m.tile_bank.get_current_size()))
        return True


def _make_vec_env():
    def _thunk():
        # Monitor gives SB3 episode stats for TensorBoard (ep_rew_mean, ep_len_mean).
        return Monitor(
            TimeLimit(
                BananaGramlEnvironment(),
                max_episode_steps=MAX_EPISODE_STEPS,
            ),
            filename=None,
        )

    return _thunk


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on BananaGramlEnvironment.")
    p.add_argument(
        "--run-name",
        default="run0",
        help="Subfolder under --tb-dir for TensorBoard; also used in default save filenames.",
    )
    p.add_argument(
        "--tb-dir",
        default="./tensorboard_logs",
        help="Root directory for TensorBoard runs (each --run-name gets a subfolder).",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=90_000,
        help="Total PPO training timesteps (default 90k).",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    p.add_argument(
        "--save-prefix",
        default=None,
        help="Prefix for bananagraml_ppo / vec_normalize saves (default: run name).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    save_prefix = args.save_prefix if args.save_prefix else args.run_name
    if args.seed is not None:
        set_random_seed(args.seed)

    tb = _tensorboard_log_dir(args.tb_dir, args.run_name)
    if tb is None and importlib.util.find_spec("tensorboard") is None:
        print(
            "Note: install tensorboard (pip install tensorboard) for live graphs: "
            "tensorboard --logdir ./tensorboard_logs",
            file=sys.stderr,
            flush=True,
        )

    venv = DummyVecEnv([_make_vec_env()])
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
        tensorboard_log=tb,
    )
    callbacks: list[BaseCallback] = [
        TrainingProgressCallback(LOG_EVERY_N_ENV_STEPS),
    ]
    if tb is not None:
        callbacks.append(TensorboardGameCallback(log_every=32))
    if not HEADLESS:
        callbacks.insert(0, LiveBoardCallback())
    learn_kw: dict = {
        "total_timesteps": args.timesteps,
        "callback": CallbackList(callbacks),
    }
    if tb is not None:
        learn_kw["tb_log_name"] = "ppo"
    model.learn(**learn_kw)
    model.save(f"{save_prefix}_ppo")
    venv.save(f"{save_prefix}_vec_normalize.pkl")
    venv.close()
    print(f"Saved {save_prefix}_ppo.zip and {save_prefix}_vec_normalize.pkl", flush=True)
    if tb:
        print(f"TensorBoard: tensorboard --logdir {os.path.abspath(args.tb_dir)}", flush=True)


if __name__ == "__main__":
    main()
