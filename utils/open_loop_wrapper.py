from gymnax.environments import environment
from gymnax.environments.spaces import Dict, Discrete
import chex
from jax import lax
import jax.numpy as jnp
from flax import struct
import jax

from collections import OrderedDict


@struct.dataclass
class EnvState:
    state: ...
    last_action: ...


class OpenLoopWrapper(environment.Environment):
    def __init__(self, env: environment.Environment):
        self.env = env

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def zero_out_obs(self, obs, state):
        t = state.state.time
        zero_obs = jnp.zeros_like(obs)
        return OrderedDict(dict(t=t, obs=zero_obs, last_action=state.last_action))

    @property
    def default_params(self):
        return self.env.default_params

    def step(self, key: chex.PRNGKey, state: EnvState, action, params):
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        # This is just a select in the default step method!
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(self, key: chex.PRNGKey, state, action, params):
        obs, state_, reward, done, info = self.env.step_env(
            key, state.state, action, params
        )

        state = state.replace(state=state_, last_action=action)

        obs = self.zero_out_obs(obs, state)
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(self, key, params):
        key, reset_key = jax.random.split(key)
        obs, state_ = self.env.reset_env(reset_key, params)
        state = EnvState(
            state=state_, last_action=self.action_space(params).sample(key)
        )
        return self.zero_out_obs(obs, state), state

    def get_obs(self, state):
        obs = self.env.get_obs(state)
        return self.zero_out_obs(obs, state)

    def is_terminal(self):
        return self.env.is_terminal()

    @property
    def name(self):
        return f"zero-out {self.env.name}"

    @property
    def num_actions(self):
        return self.env.num_actions

    def action_space(self, params):
        return self.env.action_space(params)

    def state_space(self, params):
        return self.env.state_space(params)

    def observation_space(self, params):
        return Dict(
            {
                "obs": self.env.observation_space(params),
                "t": Discrete(num_categories=params.max_steps_in_episode),
                "last_action": self.action_space(params),
            }
        )