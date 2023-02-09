import gymnax
from utils.brax_wrapper import BraxEnvironmentWrapper, BRAX_ENVS
from utils.open_loop_wrapper import OpenLoopWrapper


def make(env, zero_obs=False, **env_kwargs):
    if env in BRAX_ENVS:
        env = BraxEnvironmentWrapper(env)
        env_params = env.default_params
    else:
        env, env_params = gymnax.make(env, **env_kwargs)

    env = OpenLoopWrapper(env, zero_obs=zero_obs)
    return env, env_params
