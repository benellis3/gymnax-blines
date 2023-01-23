import jax
import wandb
from utils.models import get_model_ready
from utils.helpers import load_config, save_pkl_object


def main(
    config, mle_log, log_ext="", mask_obs=False, mask_eval=False, use_cutout=False
):
    """Run training with ES or PPO. Store logs and agent ckpt."""
    rng = jax.random.PRNGKey(config.seed_id)
    # Setup the model architecture
    rng, rng_init = jax.random.split(rng)
    model, params = get_model_ready(rng_init, config)

    # Run the training loop (either evosax ES or PPO)
    if config.train_type == "ES":
        from utils.es import train_es as train_fn
    elif config.train_type == "PPO":
        from utils.ppo import train_ppo as train_fn
    else:
        raise ValueError("Unknown train_type. Has to be in ('ES', 'PPO').")
    # Log and store the results.
    log_steps, log_return, network_ckpt = train_fn(
        rng,
        config,
        model,
        params,
        mle_log,
        mask_obs=mask_obs,
        mask_eval=mask_eval,
        use_cutout=use_cutout,
    )

    data_to_store = {
        "log_steps": log_steps,
        "log_return": log_return,
        "network": network_ckpt,
        "train_config": config,
    }

    save_pkl_object(
        data_to_store,
        f"agents/{config.env_name}/{config.train_type.lower()}{log_ext}mask_obs={mask_obs}:eval_masked={mask_eval}.pkl",
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
        "-mask",
        "--mask-obs",
        action="store_true",
        default=False,
        help="If true will randomly sample a mask for the observation",
    )
    parser.add_argument(
        "-mask-eval",
        "--mask-eval",
        action="store_true",
        default=False,
        help="Whether to mask at test time",
    )
    parser.add_argument(
        "-use-cutout",
        "--use-cutout",
        action="store_true",
        default=False,
        help="Whether to use cutout-style random masking",
    )
    parser.add_argument(
        "-no-wandb",
        "--no-wandb",
        action="store_true",
        default=False,
        help="If true will disable wandb logging",
    )

    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, args.seed_id, args.lrate)
    mode = "disabled" if args.no_wandb else "online"

    config["mask_obs"] = args.mask_obs
    config["mask_eval"] = args.mask_eval
    wandb.init(config=config, mode=mode)
    with jax.disable_jit(False):
        main(
            config.train_config,
            mle_log=None,
            log_ext=str(args.lrate) if args.lrate != 5e-04 else "",
            mask_obs=args.mask_obs,
            mask_eval=args.mask_eval,
            use_cutout=args.use_cutout,
        )
