from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from env import BananaGramlEnvironment

TOTAL_TIMESTEPS = 10_000
LOG_EVERY_N_ENV_STEPS = 10


def _base_gym_env(vec_env):
    """Unwrap DummyVecEnv to the underlying BananaGramlEnvironment."""
    envs = getattr(vec_env, "envs", None)
    if envs:
        return envs[0]
    return vec_env


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


def main():
    def make_env():
        return BananaGramlEnvironment()

    env = DummyVecEnv([make_env])
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList(
            [
                LiveBoardCallback(),
                TrainingProgressCallback(LOG_EVERY_N_ENV_STEPS),
            ]
        ),
    )
    model.save("bananagraml_ppo")
    env.close()


if __name__ == "__main__":
    main()
