from jax import lax
import jax.numpy as jnp
from brax import envs
from brax.envs import wrappers

from gymnax.environments import environment, spaces
from flax import struct

BRAX_ENVS = [
    "ant",
    "humanoid",
    "walker2d",
    "hopper",
    "halfcheetah",
    "humanoidstandup",
]

INF = 1e9


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 2


class BraxEnvironmentWrapper(environment.Environment):
    """Wraps a brax environment for use with gymnax-blines"""

    def __init__(self, env_name):
        params = self.default_params
        self.env = wrappers.AutoResetWrapper(
            wrappers.EpisodeWrapper(
                envs.get_environment(env_name=env_name),
                episode_length=params.max_steps_in_episode,
                action_repeat=1,
            )
        )

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset_env(self, key, params):
        state = self.env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params):
        action = jnp.tanh(action)
        state = self.env.step(state, action)
        return (
            lax.stop_gradient(state.obs),
            lax.stop_gradient(state),
            state.reward.astype(jnp.float32),
            state.done,
            state.info,
        )

    @property
    def default_params(self):
        return EnvParams()

    def get_obs(self, state):
        return state.obs

    def action_space(self, params):
        return spaces.Box(low=-1, high=1, shape=(self.env.action_size,))

    def observation_space(self, params):
        return spaces.Box(low=-INF, high=INF, shape=(self.env.observation_size,))

    def state_space(self, params):
        raise NotImplementedError()

    def is_terminal(self):
        raise NotImplementedError()

    @property
    def name(self):
        return f"Brax Wrapper"

    @property
    def num_actions(self) -> int:
        return self.env.action_size
