import argparse

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from env import BananaGramlEnvironment
from training_config import TrainingConfig, load_training_config


def _make_vec_env(cfg: TrainingConfig):
    def _thunk():
        return Monitor(
            TimeLimit(
                BananaGramlEnvironment(
                    render_mode=None if cfg.headless else "human",
                    starting_tiles_on_bench=cfg.starting_tiles_on_bench,
                    max_bench_tiles=cfg.max_bench_tiles,
                ),
                max_episode_steps=cfg.max_episode_steps,
            ),
            filename=None,
        )

    return _thunk


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO on BananaGraml.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training JSON (default: src/training_config.json next to training_config.py).",
    )
    args = parser.parse_args(argv)

    cfg = load_training_config(args.config)
    if cfg.random_seed is not None:
        set_random_seed(cfg.random_seed, using_cuda=False)

    venv = DummyVecEnv([_make_vec_env(cfg)])
    model = PPO("MultiInputPolicy", venv, verbose=cfg.ppo_verbose)
    model.learn(total_timesteps=cfg.total_timesteps)
    venv.close()


if __name__ == "__main__":
    main()
