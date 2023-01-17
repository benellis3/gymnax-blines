from functools import partial
from collections import namedtuple
import optax
import jax
import jax.numpy as jnp
from jax import lax
import chex
from typing import Any, Callable, Tuple
from collections import defaultdict
import flax
from flax.training.train_state import TrainState
import numpy as np
import tqdm
import gymnax
import wandb

from utils.env_mask_wrapper import ObsMaskingWrapper


def make(env, mask_obs=False, p=0.2, **env_kwargs):
    env, env_params = gymnax.make(env, **env_kwargs)
    if mask_obs:
        env = ObsMaskingWrapper(env, p=p)
    else:
        env = ObsMaskingWrapper(env, p=0)
    return env, env_params


SMALL_VALUE = -jnp.inf


class LevelManager:
    def __init__(self, size: int, state_space, sampling_method="rank"):

        self.size = size
        try:
            temp = state_space.shape[0]
            self.state_shape = state_space.shape
        except Exception:
            self.state_shape = [state_space]
        self.reset()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self):
        scores = jnp.empty((self.size,), dtype=jnp.float32)
        scores = scores.at[:].set(SMALL_VALUE)
        return {
            "levels": jnp.empty((self.size, *self.state_shape), dtype=jnp.uint8),
            "scores": scores,
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=(0,))
    def append(self, buffer, masks, scores):
        buffer, _ = lax.scan(self._append, buffer, (masks, scores))
        return buffer

    def _append(self, buffer, mask, score):
        # add the mask if it's score is in the top size elements

        min_score = jnp.min(buffer["scores"])
        min_score_idx = jnp.argmin(buffer["scores"])
        old_scores = buffer["scores"]
        old_levels = buffer["levels"]
        old_p = buffer["_p"]
        # check if not enough entries in the buffer atm
        scores = lax.cond(
            min_score == SMALL_VALUE,
            lambda: buffer["scores"].at[buffer["_p"]].set(score),
            lambda: buffer["scores"],
        )
        levels = lax.cond(
            min_score == SMALL_VALUE,
            lambda: buffer["levels"].at[buffer["_p"]].set(mask),
            lambda: buffer["levels"],
        )
        p = lax.cond(
            min_score == SMALL_VALUE, lambda: buffer["_p"] + 1, lambda: buffer["_p"]
        )
        scores = lax.cond(
            jnp.logical_and(score > min_score, buffer["_p"] == self.size),
            lambda: scores.at[min_score_idx].set(score),
            lambda: scores,
        )
        levels = lax.cond(
            jnp.logical_and(score > min_score, buffer["_p"] == self.size),
            lambda: levels.at[min_score_idx].set(mask),
            lambda: levels,
        )
        # check whether the masks are a placeholder value.
        # If so do not update anything. Needed so that we
        # don't add non-PLR levels to the buffer unless necessary
        levels = lax.cond(jnp.all(mask) == -1, lambda: old_levels, lambda: levels)
        scores = lax.cond(jnp.all(mask) == -1, lambda: old_scores, lambda: scores)
        p = lax.cond(jnp.all(mask) == -1, lambda: old_p, lambda: p)
        return {"levels": levels, "scores": scores, "_p": p}, None

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, buffer):
        # sample a level from the buffer at random
        key, subkey = jax.random.split(key)
        weights = self._generate_sample_probs(buffer)[: buffer["_p"]]
        idx = jax.random.choice(subkey, jnp.arange(buffer["_p"]), p=weights)
        return buffer["levels"][idx]

    def _generate_sample_probs(self, buffer):
        if self.sampling_method == "uniform":
            weights = jnp.ones_like(buffer["scores"])
        elif self.sampling_method == "rank":
            scores = jnp.where(
                np.arange(self.size) < buffer["_p"], buffer["scores"], jnp.inf
            )
            ranks = jnp.argsort(scores)
            weights = jnp.where(np.arange(self.size) < buffer["_p"], 1.0 / ranks, 0)
        elif self.sampling_method == "power":
            weights = jnp.clip(buffer["scores"], a_min=0)

        return weights

    def compute_value_loss(self, returns: chex.Array, value_preds: chex.Array):
        clipped_advantages = jnp.clip((returns - value_preds), a_min=0)
        return jnp.mean(clipped_advantages)


Batch = namedtuple(
    "Batch", ["state", "actions", "log_pis_old", "values_old", "target", "gae"]
)


class BatchManager:
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        n_steps: int,
        num_envs: int,
        action_size,
        state_space,
    ):
        self.num_envs = num_envs
        self.action_size = action_size
        self.buffer_size = num_envs * n_steps
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.discount = discount
        self.gae_lambda = gae_lambda

        try:
            temp = state_space.shape[0]
            self.state_shape = state_space.shape
        except Exception:
            self.state_shape = [state_space]
        self.reset()

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        return {
            "states": jnp.empty(
                (self.n_steps, self.num_envs, *self.state_shape),
                dtype=jnp.float32,
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_envs, *self.action_size),
            ),
            "rewards": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "dones": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.uint8),
            "log_pis_old": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "values_old": jnp.empty((self.n_steps, self.num_envs), dtype=jnp.float32),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, state, action, reward, done, log_pi, value):
        return {
            "states": buffer["states"].at[buffer["_p"]].set(state),
            "actions": buffer["actions"].at[buffer["_p"]].set(action),
            "rewards": buffer["rewards"].at[buffer["_p"]].set(reward.squeeze()),
            "dones": buffer["dones"].at[buffer["_p"]].set(done.squeeze()),
            "log_pis_old": buffer["log_pis_old"].at[buffer["_p"]].set(log_pi),
            "values_old": buffer["values_old"].at[buffer["_p"]].set(value),
            "_p": (buffer["_p"] + 1) % self.n_steps,
        }

    @partial(jax.jit, static_argnums=0)
    def get(self, buffer):
        gae, target = self.calculate_gae(
            value=buffer["values_old"],
            reward=buffer["rewards"],
            done=buffer["dones"],
        )
        batch = Batch(
            buffer["states"][:-1],
            buffer["actions"][:-1],
            buffer["log_pis_old"][:-1],
            buffer["values_old"][:-1],
            target,
            gae,
        )
        return batch

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = self.discount * value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + self.discount * self.gae_lambda * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]


def split_n_keys(keys):
    keys, subkeys = jax.vmap(jax.random.split)(keys)
    return keys, subkeys


class RolloutManager(object):
    def __init__(
        self,
        model,
        env_name,
        env_kwargs,
        env_params,
        eval_env_names=None,
        mask_obs=False,
        p=0,
        use_plr=False,
        plr_prob=0.5,
        **plr_kwargs,
    ):
        # Setup functionalities for vectorized batch rollout
        self.env_name = env_name
        self.env, self.env_params = make(env_name, mask_obs=mask_obs, p=p, **env_kwargs)
        self.env_params = self.env_params.replace(**env_params)

        if not eval_env_names:
            self.eval_envs = [
                self.env,
            ]
            self.eval_env_params = [self.env_params]
        else:
            self.eval_envs = []
            self.eval_env_params = []
            for eval_env_name in zip(eval_env_names):
                eval_env, eval_env_params = make(eval_env_name, mask_obs=False, p=0)
                self.eval_envs.append(eval_env)
                self.eval_env_params.append(eval_env_params)
        self.observation_space = self.env.observation_space(self.env_params)
        self.action_size = self.env.action_space(self.env_params).shape
        self.apply_fn = model.apply
        self.select_action = self.select_action_ppo
        self.use_plr = use_plr
        self.plr_prob = plr_prob
        if self.use_plr:
            self.level_manager = LevelManager(**plr_kwargs)
            self.level_buffer = self.level_manager.reset()

    @partial(jax.jit, static_argnums=0)
    def select_action_ppo(
        self,
        train_state: TrainState,
        obs: jnp.ndarray,
        rng: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.random.PRNGKey]:
        value, pi = policy(train_state.apply_fn, train_state.params, obs, rng)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value[:, 0], rng

    @partial(jax.jit, static_argnums=(0, 2))
    def batch_reset(self, keys, evaluate=False):
        # choose which envs will be masked and which won't
        if self.use_plr and not evaluate:
            keys, subkeys = split_n_keys(keys)
            obs_re, state_re = jax.vmap(self.env.reset, in_axes=(0, None))(
                subkeys, self.env_params
            )
            keys, subkeys = split_n_keys(keys)
            masks = jax.vmap(self.level_buffer.sample, in_axes=(0, None))(
                subkeys, self.env_params
            )

            keys, subkeys = split_n_keys(keys)
            obs_plr, state_plr = jax.vmap(self.env.reset_mask, in_axes=(0, None, 0))(
                subkeys, self.env_params, masks
            )
            # sample which ones will be reset which way
            plr_mask = jax.random.choice(
                keys[0],
                jnp.arange(2),
                shape=(keys.shape[0],),
                p=jnp.array([1.0 - self.plr_prob, self.plr_prob]),
            )
            obs = jnp.where(plr_mask > 0, obs_plr, obs_re)
            state = jnp.where(plr_mask > 0, state_plr, state_re)

        else:
            obs, state = jax.vmap(self.env.reset, in_axes=(0, None))(
                jnp.array(keys), self.env_params
            )
            plr_mask = jnp.ones(shape=(len(keys),))
            masks = jnp.zeros_like(obs)
        return plr_mask, masks, obs, state

    @partial(jax.jit, static_argnums=(0,))
    def batch_step(self, keys, state, action):
        return self._batch_step(self.env, self.env_params, keys, state, action)

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def _batch_step(self, env, env_params, keys, state, action):
        if self.use_plr:
            return jax.vmap(env.step_mask, in_axes=(0, 0, 0, None))(
                keys, state, action, env_params
            )
        else:
            return jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                keys, state, action, env_params
            )

    def update_level_buffer(self, returns, value_preds, masks, plr_mask):
        regrets = jax.vmap(self.level_manager.compute_value_loss)(returns, value_preds)
        buffer_add_mask = jnp.logical_not(plr_mask)
        masks = jnp.where(buffer_add_mask > 0, masks, -jnp.ones_like(masks))
        scores = jnp.where(buffer_add_mask > 0, regrets, SMALL_VALUE)
        # when the masks are -1, the append operation does nothing
        self.buffer = self.level_manager.append(self.buffer, masks, scores)

    @partial(jax.jit, static_argnums=(0, 3))
    def batch_evaluate(self, rng_input, train_state, num_envs):
        """Evaluate all the eval_envs using num_envs environments
        in parallel for an episode"""
        results = {}
        for env, env_params in zip(self.eval_envs, self.eval_env_params):
            rng_input, sub_rng = jax.random.split(rng_input)
            cum_return = self._batch_evaluate(
                env, env_params, sub_rng, train_state, num_envs
            )
            results[f"{env.name}:cum_return"] = cum_return
        return results

    @partial(jax.jit, static_argnums=(0, 1, 2, 5))
    def _batch_evaluate(self, env, env_params, rng_input, train_state, num_envs):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        _, _, obs, state = self.batch_reset(
            jax.random.split(rng_reset, num_envs), evaluate=True
        )

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, train_state, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action, _, _, rng = self.select_action(train_state, obs, rng_net)
            next_o, next_s, reward, done, _ = self._batch_step(
                env,
                env_params,
                jax.random.split(rng_step, num_envs),
                state,
                action.squeeze(),
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry, y = [
                next_o,
                next_s,
                train_state,
                rng,
                new_cum_reward,
                new_valid_mask,
            ], [new_valid_mask]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                train_state,
                rng_episode,
                jnp.array(num_envs * [0.0]),
                jnp.array(num_envs * [1.0]),
            ],
            (),
            env_params.max_steps_in_episode,
        )

        cum_return = carry_out[-2].squeeze()
        return jnp.mean(cum_return)


@partial(jax.jit, static_argnums=0)
def policy(
    apply_fn: Callable[..., Any],
    params: flax.core.frozen_dict.FrozenDict,
    obs: jnp.ndarray,
    rng,
):
    value, pi = apply_fn(params, obs, rng)
    return value, pi


def train_ppo(rng, config, model, params, mle_log, mask_obs=False):
    """Training loop for PPO based on https://github.com/bmazoure/ppo_jax."""
    num_total_epochs = int(config.num_train_steps // config.num_train_envs + 1)
    num_steps_warm_up = int(config.num_train_steps * config.lr_warmup)
    schedule_fn = optax.linear_schedule(
        init_value=-float(config.lr_begin),
        end_value=-float(config.lr_end),
        transition_steps=num_steps_warm_up,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(schedule_fn),
    )

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    # Setup the rollout manager -> Collects data in vmapped-fashion over envs
    rollout_manager = RolloutManager(
        model,
        config.env_name,
        config.env_kwargs,
        config.env_params,
        mask_obs=mask_obs,
        p=config.p,
        use_plr=config.use_plr,
        plr_prob=config.plr_prob,
    )

    batch_manager = BatchManager(
        discount=config.gamma,
        gae_lambda=config.gae_lambda,
        n_steps=config.n_steps + 1,
        num_envs=config.num_train_envs,
        action_size=rollout_manager.action_size,
        state_space=rollout_manager.observation_space,
    )

    @partial(jax.jit, static_argnums=5)
    def get_transition(
        train_state: TrainState,
        obs: jnp.ndarray,
        state: dict,
        batch,
        rng: jax.random.PRNGKey,
        num_train_envs: int,
    ):
        action, log_pi, value, new_key = rollout_manager.select_action(
            train_state, obs, rng
        )
        # print(action.shape)
        new_key, key_step = jax.random.split(new_key)
        b_rng = jax.random.split(key_step, num_train_envs)
        # Automatic env resetting in gymnax step!
        next_obs, next_state, reward, done, _ = rollout_manager.batch_step(
            b_rng, state, action
        )
        batch = batch_manager.append(batch, obs, action, reward, done, log_pi, value)
        return train_state, next_obs, next_state, batch, new_key

    batch = batch_manager.reset()

    rng, rng_step, rng_reset, rng_eval, rng_update = jax.random.split(rng, 5)
    rng_reset, *brngs = jax.random.split(rng_reset, config.num_train_envs + 1)
    brngs = jnp.array(brngs)
    plr_mask, masks, obs, state = rollout_manager.batch_reset(brngs)

    total_steps = 0
    log_steps, log_return = [], []
    updates = 0
    t = tqdm.tqdm(range(1, num_total_epochs + 1), desc="PPO", leave=True)
    for step in t:
        train_state, obs, state, batch, rng_step = get_transition(
            train_state,
            obs,
            state,
            batch,
            rng_step,
            config.num_train_envs,
        )
        total_steps += config.num_train_envs
        if step % (config.n_steps + 1) == 0:
            batch = batch_manager.get(batch)
            metric_dict, train_state, rng_update = update(
                train_state,
                batch,
                config.num_train_envs,
                config.n_steps,
                config.n_minibatch,
                config.epoch_ppo,
                config.clip_eps,
                config.entropy_coeff,
                config.critic_coeff,
                rng_update,
                plr_mask=plr_mask,
            )
            updates = updates + 1
            # wandb.log(metric_dict)
            if config.use_plr:
                rollout_manager.update_level_buffer(
                    batch.target, batch.values_old, masks, plr_mask
                )
            batch = batch_manager.reset()
            if config.use_plr:
                rng_reset, *brngs = jax.random.split(rng_reset, config.num_train_envs + 1)
                brngs = jnp.array(brngs)
                plr_mask, masks, obs, state = rollout_manager.batch_reset(brngs)

        if (step + 1) % config.evaluate_every_epochs == 0:
            rng, rng_eval = jax.random.split(rng)
            results = rollout_manager.batch_evaluate(
                rng_eval,
                train_state,
                config.num_test_rollouts,
            )
            rewards = jnp.array(list(results.values())).mean()
            log_steps.append(total_steps)
            log_return.append(rewards)
            t.set_description(f"R: {str(rewards)}")
            t.refresh()
            wandb.log({"steps": total_steps, "updates": updates, **results})
            if mle_log is not None:
                mle_log.update(
                    {"num_steps": total_steps},
                    {"return": rewards},
                    model=train_state.params,
                    save=True,
                )

    return (
        log_steps,
        log_return,
        train_state.params,
    )


@jax.jit
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def loss_actor_and_critic(
    params_model: flax.core.frozen_dict.FrozenDict,
    apply_fn: Callable[..., Any],
    obs: jnp.ndarray,
    target: jnp.ndarray,
    value_old: jnp.ndarray,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    plr_mask: jnp.ndarray,
    action: jnp.ndarray,
    clip_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> jnp.ndarray:

    value_pred, pi = apply_fn(params_model, obs, rng=None)
    value_pred = value_pred[:, 0]

    # TODO: Figure out why training without 0 breaks categorical model
    # And why with 0 breaks gaussian model pi
    log_prob = pi.log_prob(action[..., -1])

    value_pred_clipped = value_old + (value_pred - value_old).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_losses = 0.5 * jnp.maximum(value_losses, value_losses_clipped)
    value_losses = jnp.where(plr_mask > 0, value_losses, 0.0)
    value_loss = value_losses.sum() / plr_mask.sum()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae_mean = gae.sum() / plr_mask.sum()
    gae = (gae - gae_mean) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = jnp.where(plr_mask > 0, loss_actor, 0.0)
    loss_actor = loss_actor.sum() / plr_mask.sum()

    entropy = jnp.where(plr_mask > 0, pi.entropy(), 0)
    entropy = entropy.sum() / plr_mask.sum()

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * entropy

    return total_loss, (
        value_loss,
        loss_actor,
        entropy,
        value_pred.sum() / plr_mask.sum(),
        target.sum() / plr_mask.sum(),
        gae_mean,
    )


def update(
    train_state: TrainState,
    batch: Tuple,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    epoch_ppo: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    rng: jax.random.PRNGKey,
    plr_mask: chex.Array,
):
    """Perform multiple epochs of updates with multiple updates."""
    obs, action, log_pi_old, value, target, gae = batch
    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch
    idxes = jnp.arange(num_envs * n_steps)
    plr_mask = jnp.repeat(jnp.expand_dims(plr_mask, -1), n_steps, axis=1).reshape((-1,))
    avg_metrics_dict = defaultdict(int)
    for _ in range(epoch_ppo):
        idxes = jax.random.permutation(rng, idxes)
        idxes_list = [
            idxes[start : start + size_minibatch]
            for start in jnp.arange(0, size_batch, size_minibatch)
        ]

        train_state, total_loss = update_epoch(
            train_state,
            idxes_list,
            flatten_dims(obs),
            flatten_dims(action),
            flatten_dims(log_pi_old),
            flatten_dims(value),
            jnp.array(flatten_dims(target)),
            jnp.array(flatten_dims(gae)),
            clip_eps,
            entropy_coeff,
            critic_coeff,
            plr_mask,
        )

        total_loss, (
            value_loss,
            loss_actor,
            entropy,
            value_pred,
            target_val,
            gae_val,
        ) = total_loss

        avg_metrics_dict["total_loss"] += np.asarray(total_loss)
        avg_metrics_dict["value_loss"] += np.asarray(value_loss)
        avg_metrics_dict["actor_loss"] += np.asarray(loss_actor)
        avg_metrics_dict["entropy"] += np.asarray(entropy)
        avg_metrics_dict["value_pred"] += np.asarray(value_pred)
        avg_metrics_dict["target"] += np.asarray(target_val)
        avg_metrics_dict["gae"] += np.asarray(gae_val)

    for k, v in avg_metrics_dict.items():
        avg_metrics_dict[k] = v / (epoch_ppo)

    return avg_metrics_dict, train_state, rng


@jax.jit
def update_epoch(
    train_state: TrainState,
    idxes: jnp.ndarray,
    obs,
    action,
    log_pi_old,
    value,
    target,
    gae,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    plr_mask: chex.Array,
):
    for idx in idxes:
        # print(action[idx].shape, action[idx].reshape(-1, 1).shape)
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params,
            train_state.apply_fn,
            obs=obs[idx],
            target=target[idx],
            value_old=value[idx],
            log_pi_old=log_pi_old[idx],
            gae=gae[idx],
            plr_mask=plr_mask[idx],
            # action=action[idx].reshape(-1, 1),
            action=jnp.expand_dims(action[idx], -1),
            clip_eps=clip_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
        )
        train_state = train_state.apply_gradients(grads=grads)
    return train_state, total_loss
