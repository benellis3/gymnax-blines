import numpy as np
import jax
import gymnax
from gymnax.visualize import Visualizer
from utils.models import get_model_ready
from utils.helpers import load_pkl_object, load_config
from utils.masked_visualizer import MaskedVisualizer


def load_neural_network(config, agent_path):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config)

    params = load_pkl_object(agent_path)["network"]
    return model, params


def rollout_episode(env, env_params, model, model_params, max_frames=300, seed=0):
    state_seq = []
    rng = jax.random.PRNGKey(seed)

    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)

    if model is not None:
        if model.model_name == "LSTM":
            hidden = model.initialize_carry()

    t_counter = 0
    reward_seq = []
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            if model.model_name == "LSTM":
                hidden, action = model.apply(model_params, obs, hidden, rng_act)
            else:
                if model.model_name.startswith("separate"):
                    v, pi = model.apply(model_params, obs, rng_act)
                    action = pi.sample(seed=rng_act)
                else:
                    action = model.apply(model_params, obs, rng_act)
        else:
            action = 0  # env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        print(t_counter, reward, action, done)
        print(10 * "=")
        t_counter += 1
        if done or t_counter == max_frames:
            break
        else:
            env_state = next_env_state
            obs = next_obs
    print(f"{env.name} - Steps: {t_counter}, Return: {np.sum(reward_seq)}")
    return state_seq, np.cumsum(reward_seq)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train",
        "--train_type",
        type=str,
        default="ppo",
        help="es/ppo trained agent.",
    )
    parser.add_argument(
        "-model",
        "--model_name",
        type=str,
        default="",
        help="File name of the model",
    )
    parser.add_argument(
        "-random",
        "--random",
        action="store_true",
        default=False,
        help="Random rollout.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Environment name.",
    )
    parser.add_argument(
        "-p",
        "--mask_prob",
        type=float,
        default=0,
        help="Mask probability",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=False,
        help="View the gif",
    )
    parser.add_argument(
        "-cutout",
        "--use_cutout",
        action="store_true",
        default=False,
        help="Whether to use cutout-style random masking",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Rollout seed",
    )

    args, _ = parser.parse_known_args()

    base = f"agents/{args.env_name}/{args.train_type}"
    configs = load_config(base + ".yaml")
    if len(args.model_name) > 0:
        model = f"agents/{args.env_name}/{args.model_name}"
    else:
        model = base

    if not args.random:
        model, model_params = load_neural_network(
            configs.train_config, model + ".pkl"
        )
    else:
        model, model_params = None, None

    env, env_params = gymnax.make(
        configs.train_config.env_name,
        **configs.train_config.env_kwargs,
    )
    if args.mask_prob >= 0:
        from utils.env_mask_wrapper import ObsMaskingWrapper
        env = ObsMaskingWrapper(env, use_cutout=args.use_cutout, p=args.mask_prob)

    env_params.replace(**configs.train_config.env_params)
    state_seq, cum_rewards = rollout_episode(
        env, env_params, model, model_params, seed=args.seed
    )
    if hasattr(env, "is_masked") and env.is_masked:
        vis = MaskedVisualizer(env, env_params, state_seq, cum_rewards)
    else:
        vis = Visualizer(env, env_params, state_seq, cum_rewards)
    # vis.animate(f"docs/{args.env_name}-p{args.mask_prob}.gif", view=args.view)
    gif_name = f"docs/{args.env_name}-{args.model_name}"
    if args.use_cutout:
        gif_name += f"_cutout_seed{args.seed}.gif"
    else:
        gif_name += f"_p{args.mask_prob}_seed{args.seed}.gif"
    vis.animate(gif_name, view=args.view)