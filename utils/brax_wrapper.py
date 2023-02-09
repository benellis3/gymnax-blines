from jax import lax
import jax.numpy as jnp
from brax import envs

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
    max_steps_in_episode: int = 1000


class BraxEnvironmentWrapper(environment.Environment):
    """Wraps a brax environment for use with gymnax-blines"""

    def __init__(self, env_name):
        self.env = envs.get_environment(env_name=env_name)

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset_env(self, key, params):
        state = self.env.reset(key)
        state.info["t"] = 0
        return state.obs, state

    def step_env(self, key, state, action, params):
        # apply tanh to that bad boy to force it to be between -1 and 1
        action = jnp.tanh(action)
        state = self.env.step(state=state, action=action)
        reward = state.reward
        state.info["t"] = state.info["t"] + 1
        done = jnp.where(
            state.info["t"] >= params.max_steps_in_episode, True, state.done
        )
        obs = state.obs
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done.astype(jnp.bool_),
            {},
        )

    @property
    def default_params(self):
        return EnvParams()

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
