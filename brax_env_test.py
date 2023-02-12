"""Environment test for brax"""
from datetime import datetime
from pprint import pprint
from brax import jumpy as jp
from brax.training.agents.ppo import train as ppo
from brax import envs
from brax.io import model
import jax
from flax.training.train_state import TrainState

import functools
from matplotlib import pyplot as plt
import wandb
from utils.ppo import RolloutManager


def evaluate_on_gymnax(path, inference_fn, env_name, rng, num_test_rollouts=164):
    params = model.load_params(path)
    rng, rng_eval = jax.random.split(rng)
    train_state = TrainState.create(apply_fn=inference_fn, params=params, tx=None)
    rollout_manager = RolloutManager(
        model=None,
        zero_obs=False,
        clamp_action=False,
        env_name=env_name,
        env_kwargs={},
        env_params={},
    )
    rewards = rollout_manager.batch_evaluate(rng_eval, train_state, num_test_rollouts)
    print(f"Mean Rewards: {rewards}")


def train():
    env_name = "ant"  # @param ['ant', 'fetch', 'grasp', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'pusher', 'reacher', 'walker2d', 'grasp', 'ur5e']
    env = envs.get_environment(env_name=env_name)
    state = env.reset(rng=jp.random_prngkey(seed=0))
    train_fn = {
        # OpenAI gym environments:
        "ant": functools.partial(
            ppo.train,
            num_timesteps=25000000,
            num_evals=20,
            reward_scaling=10,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=5,
            num_minibatches=32,
            num_updates_per_batch=4,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
        ),
        "halfcheetah": functools.partial(
            ppo.train,
            num_timesteps=100_000_000,
            num_evals=20,
            reward_scaling=1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=20,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.95,
            learning_rate=3e-4,
            entropy_cost=0.001,
            num_envs=2048,
            batch_size=512,
        ),
        "humanoid": functools.partial(
            ppo.train,
            num_timesteps=50_000_000,
            num_evals=20,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=10,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.97,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            num_envs=2048,
            batch_size=1024,
            seed=1,
        ),
        "humanoidstandup": functools.partial(
            ppo.train,
            num_timesteps=100_000_000,
            num_evals=20,
            reward_scaling=0.1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            unroll_length=15,
            num_minibatches=32,
            num_updates_per_batch=8,
            discounting=0.97,
            learning_rate=6e-4,
            entropy_cost=1e-2,
            num_envs=2048,
            batch_size=1024,
        ),
    }[env_name]
    max_y = {
        "ant": 8000,
        "halfcheetah": 8000,
        "hopper": 2500,
        "humanoid": 13000,
        "humanoidstandup": 75_000,
        "reacher": 5,
        "walker2d": 5000,
        "fetch": 15,
        "grasp": 100,
        "ur5e": 10,
        "pusher": 0,
    }[env_name]
    min_y = {"reacher": -100, "pusher": -150}.get(env_name, 0)
    xdata, ydata = [], []
    times = [datetime.now()]
    wandb.init(project="brax-training")

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"])
        wandb.log(metrics)
        # plt.xlim([0, train_fn.keywords["num_timesteps"]])
        # plt.ylim([min_y, max_y])
        # plt.xlabel("# environment steps")
        # plt.ylabel("reward per episode")
        # plt.plot(xdata, ydata)
        # plt.show()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
    path = "/Users/benellis/brax_models"
    model.save_params(path, params)
    pprint(ydata)
    evaluate_on_gymnax(
        path=path,
        inference_fn=make_inference_fn(params),
        env_name="ant",
        rng=jax.random.PRNGKey(0),
    )


if __name__ == "__main__":
    train()
