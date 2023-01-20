import jax
import jax.numpy as jnp
from jax import lax
import chex
from flax import struct
from gymnax.environments import environment
from functools import partial


@struct.dataclass
class EnvState:
    state: ...
    mask: chex.Array


class ObsMaskingWrapper(environment.Environment):
    def __init__(self, env: environment.Environment, use_cutout: bool, p: float):
        self.env = env
        self.use_cutout = use_cutout
        self.p = p
        self.is_masked = p > 0

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    @property
    def default_params(self):
        return self.env.default_params

    def step_mask(self, key: chex.PRNGKey, state: EnvState, action, params):
        """Version of step from environment.Environment that uses a consistent
        mask when resetting"""
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env_mask(key_reset, params, state.mask)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

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

    def _sample_mask(self, key):
        if self.use_cutout:
            # sample a random 3x3 cutout of the observation
            key, subkey = jax.random.split(key)
            i = jax.random.randint(
                subkey, shape=(), minval=1, maxval=self.obs_shape[0] + 1
            )
            key, subkey = jax.random.split(key)
            j = jax.random.randint(
                subkey, shape=(), minval=1, maxval=self.obs_shape[1] + 1
            )
            mask = jnp.zeros(
                shape=(
                    self.obs_shape[0] + 2,
                    self.obs_shape[1] + 2,
                    *self.obs_shape[2:],
                ),
                dtype=jnp.uint8,
            )
            mask = jax.lax.dynamic_update_slice(
                mask,
                jnp.ones((3, 3, *self.obs_shape[2:]), dtype=jnp.uint8),
                (i - 1, j - 1, 0),
            )
        else:
            key, subkey = jax.random.split(key)
            shape = (self.obs_shape[0] + 2, self.obs_shape[1] + 2, *self.obs_shape[2:])
            mask = jax.random.choice(
                subkey,
                jnp.arange(2),
                shape=shape,
                replace=True,
                p=jnp.array([1.0 - self.p, self.p]),
            )

        return mask

    def reset_env(self, key, params):
        mask = self._sample_mask(key)
        return self.reset_env_mask(key, params, mask)

    @partial(jax.jit, static_argnums=(0,))
    def reset_mask(self, key, params, mask):
        """A version of the reset wrapper from
        gymnax that resets to the given mask when the
        episode is done"""
        if params is None:
            params = self.default_params
        obs, state = self.reset_env_mask(key, params, mask)
        return obs, state

    def reset_env_mask(self, key, params, mask):
        obs, state = self.env.reset_env(key, params)
        state = EnvState(state=state, mask=mask)
        return self._mask_obs(obs, state), state

    def get_obs(self, state):
        return self._mask_obs(self.env.get_obs(state.state), state)

    def _mask_obs(self, obs: chex.Array, state: EnvState):
        return jnp.where(
            state.mask[1:-1, 1:-1] > 0,
            -jnp.ones(shape=self.obs_shape, dtype=obs.dtype),
            obs,
        )

    def get_mask(self, state):
        return state.mask

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
