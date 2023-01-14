import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct
from gymnax.environments import environment


@struct.dataclass
class EnvState:
    state: ...
    mask: chex.Array


class ObsMaskingWrapper(environment.Environment):
    def __init__(self, env: environment.Environment, p: float):
        self.env = env
        self.p = p

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    @property
    def default_params(self):
        return self.env.default_params

    def step_env(self, key: chex.PRNGKey, state: EnvState, action, params):
        obs, state_, reward, done, info = self.env.step_env(
            key, state.state, action, params
        )
        obs = self._mask_obs(obs, state)

        state = state.replace(state=state_)
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(self, key, params):
        key, subkey = jax.random.split(key)
        obs, state = self.env.reset_env(subkey, params)
        key, subkey = jax.random.split(key)
        mask = jax.random.choice(
            subkey,
            jnp.arange(2),
            shape=self.obs_shape,
            replace=True,
            p=jnp.array([1.0 - self.p, self.p]),
        )
        state = EnvState(state=state, mask=mask)
        return self._mask_obs(obs, state), state

    def get_obs(self, state):
        return self._mask_obs(self.env.get_obs(state))

    def _mask_obs(self, obs: chex.Array, state: EnvState):
        return jnp.where(
            state.mask > 0, jnp.zeros(shape=self.obs_shape, dtype=obs.dtype), obs
        )

    def is_terminal(self):
        return self.env.is_terminal()

    @property
    def name(self):
        return self.env.name

    @property
    def num_actions(self):
        return self.env.num_actions

    def action_space(self, params):
        return self.env.action_space(params)

    def state_space(self, params):
        # This isn't used anywhere currently so I haven't
        # modified the original. Need to think about
        # correctness potentially here though
        return self.env.state_space(params)

    def observation_space(self, params):
        return self.env.observation_space(params)
