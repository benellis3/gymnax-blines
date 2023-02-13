import jax
from utils.models import get_model_ready
from utils.helpers import load_config, save_pkl_object
import wandb


def main(config, mle_log, log_ext="", zero_obs=False, first_obs=False):
    """Run training with ES or PPO. Store logs and agent ckpt."""
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)
    model, params = get_model_ready(rng_init, config, zero_obs=zero_obs)

    # Run the training loop (either evosax ES or PPO)
    if config.train_type == "ES":
        from utils.es import train_es as train_fn
    elif config.train_type == "PPO":
        from utils.ppo import train_ppo as train_fn
    else:
        raise ValueError("Unknown train_type. Has to be in ('ES', 'PPO').")

    # Log and store the results.
    log_steps, log_return, network_ckpt = train_fn(
        rng, config, model, params, mle_log, zero_obs=zero_obs, first_obs=first_obs
    )

    data_to_store = {
        "log_steps": log_steps,
        "log_return": log_return,
        "network": network_ckpt,
        "train_config": config,
    }

    save_pkl_object(
        data_to_store,
        f"agents/{config.env_name}/{config.train_type.lower()}{log_ext}.pkl",
    )


if __name__ == "__main__":
    # Use MLE-Infrastructure if available (e.g. for parameter search)
    # try:
    #     from mle_toolbox import MLExperiment

    #     mle = MLExperiment(config_fname="configs/cartpole/ppo.yaml")
    #     main(mle.train_config, mle_log=mle.log)
    # # Otherwise use simple logging and config loading
    # except Exception:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="configs/CartPole-v1/ppo.yaml",
        help="Path to configuration yaml.",
    )
    parser.add_argument(
        "-seed",
        "--seed_id",
        type=int,
        default=0,
        help="Random seed of experiment.",
    )
    parser.add_argument(
        "-lr",
        "--lrate",
        type=float,
        default=5e-04,
        help="Random seed of experiment.",
    )
    parser.add_argument(
        "-no-wandb",
        "--no-wandb",
        action="store_true",
        default=False,
        help="If true will disable wandb logging",
    )
    parser.add_argument(
        "-zero-obs",
        "--zero-obs",
        action="store_true",
        default=False,
        help="Whether to zero out observations",
    )
    parser.add_argument(
        "-first-obs",
        "--first-obs",
        action="store_true",
        default=False,
        help="Whether to just pass the first observation"
    )

    args, _ = parser.parse_known_args()
    mode = "disabled" if args.no_wandb else "online"
    config = load_config(args.config_fname, args.seed_id, args.lrate)
    wandb.init(config=config, mode=mode)
    with jax.disable_jit(False):
        main(
            config.train_config,
            mle_log=None,
            log_ext=str(args.lrate) if args.lrate != 5e-04 else "",
            zero_obs=args.zero_obs,
            first_obs=args.first_obs
        )
