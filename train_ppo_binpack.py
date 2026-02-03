#!/usr/bin/env python3
"""
PPO Training for 3D Bin Packing
===============================

A clean implementation of PPO for the Jumanji 3D BinPack environment. 
This serves as a baseline comparison to our Expert Iteration / AlphaZero-style approach.

Why compare PPO vs Expert Iteration?
------------------------------------
PPO learns from sampled actions and scalar rewards. Expert Iteration learns
from full MCTS action distributions. The difference: Expert Iteration gets
a much richer training signal at each step, which is why it tends to outperform
PPO on planning-heavy tasks like bin packing.

That said, PPO is simpler (no search at training time) and often a solid baseline.

Key PPO concepts implemented here:
  - Clipped surrogate objective (prevents too-large policy updates)
  - Generalized Advantage Estimation (GAE) for variance reduction
  - Value function clipping (optional but helps stability)
  - Entropy bonus (encourages exploration)

References:
  - PPO paper: https://arxiv.org/abs/1707.06347
  - GAE paper: https://arxiv.org/abs/1506.02438
  - Jumanji BinPack: https://github.com/instadeepai/jumanji
  - An awesome clean jax PPO implementation: https://github.com/luchris429/purejaxrl
  - 37 Implementation Details of PPO: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

Usage:
  python train_ppo_binpack.py
  python train_ppo_binpack.py --seed 42 learning_rate=1e-4
  python train_ppo_binpack.py clip_eps=0.1 num_epochs=8
"""

import argparse
import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import wandb
from omegaconf import OmegaConf
from pydantic import BaseModel


# =============================================================================
# Config
# =============================================================================
# Pydantic for type-safe config, OmegaConf for CLI overrides like `key=value`

class Config(BaseModel):
    # env
    env_id: str = "BinPack-v2"
    seed: int = 0

    # training loop
    max_num_iters: int = 800
    eval_interval: int = 20
    save_interval: int = 200
    save_dir: str = "./checkpoints"

    # rollout
    rollout_batch_size: int = 1024    # episodes per iteration (across all devices)
    max_num_steps: int = 20           # BinPack-v2 has up to 20 items
    discount: float = 1.0             # undiscounted for episodic tasks

    # ppo hp
    num_epochs: int = 4               # epochs per iteration over collected data
    clip_eps: float = 0.2             # PPO clipping threshold
    gae_lambda: float = 0.95          # GAE lambda for advantage estimation
    entropy_coef: float = 0.01        # entropy bonus weight
    value_coef: float = 0.5           # value loss weight
    value_clip_eps: float = 0.2       # value clipping (set large to disable)
    max_grad_norm: float = 1.0        # gradient clipping

    # sgd
    training_batch_size: int = 4096   # samples per update (across devices)
    learning_rate: float = 3e-4

    # nn architecture (same as Expert Iteration for fair comparison)
    num_transformer_layers: int = 4
    transformer_num_heads: int = 4
    transformer_key_size: int = 32
    transformer_mlp_units: Sequence[int] = (256, 256)

    # eval
    eval_batch_size: int = 1024

    # logging
    wandb_project: str = "jumanji-binpack"
    run_name_prefix: str = "PPO"

    # compatibility (ignored, lets you reuse scripts that pass --num_simulations)
    num_simulations: int = 0

    class Config:
        extra = "forbid"


# -----------------------------------------------------------------------------
# cli
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="PPO for 3D Bin Packing")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_simulations", type=int, default=None, help="Ignored (compat)")
args, unknown = parser.parse_known_args()

conf_dict = OmegaConf.to_container(OmegaConf.from_cli(unknown), resolve=True) or {}
if args.seed is not None:
    conf_dict["seed"] = args.seed
if args.num_simulations is not None:
    conf_dict["num_simulations"] = args.num_simulations

config = Config(**conf_dict)
print("=" * 60)
print("Config:")
print("=" * 60)
print(config)
print()


# =============================================================================
# JAX devices
# =============================================================================
# jax.pmap distributes computation across all available devices (GPUs/TPUs)
# Each device handles (batch_size / num_devices) samples independently
# Gradients are synced with jax.lax.pmean

devices = jax.local_devices()
num_devices = len(devices)
print(f"JAX devices: {devices}")
print(f"num_devices: {num_devices}")

assert config.rollout_batch_size % num_devices == 0
assert config.training_batch_size % num_devices == 0
assert config.eval_batch_size % num_devices == 0
print()


# =============================================================================
# Env
# =============================================================================
# BinPack action space is MultiDiscrete([obs_num_ems, max_num_items])
# We flatten to a single discrete action for simpler handling

import jumanji

env = jumanji.make(config.env_id)

obs_num_ems = int(getattr(env, "obs_num_ems", env.action_spec.num_values[0]))
max_num_items = int(
    getattr(getattr(env, "generator", None), "max_num_items", env.action_spec.num_values[1])
)
num_actions = obs_num_ems * max_num_items

print(f"Environment: {config.env_id}")
print(f"  obs_num_ems: {obs_num_ems}")
print(f"  max_num_items: {max_num_items}")
print(f"  num_actions (flattened): {num_actions}")
print()

# large negative for masking (not -inf to avoid NaN in softmax)
NEG_INF = jnp.array(-1e9, dtype=jnp.float32)


# =============================================================================
# Action utilities
# =============================================================================

def unflatten_action(action: jnp.ndarray) -> jnp.ndarray:
    """Flat action index → (ems_id, item_id). Shape: (B,) → (B, 2)"""
    ems_id = action // max_num_items
    item_id = action % max_num_items
    return jnp.stack([ems_id, item_id], axis=-1).astype(jnp.int32)


def get_valid_action_mask(action_mask_2d: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten 2D action mask and ensure at least one valid action.
    
    action_mask_2d: (B, E, I) bool
    returns: (B, A) bool with ≥1 True per row
    """
    flat = action_mask_2d.reshape((action_mask_2d.shape[0], -1))
    has_any = jnp.any(flat, axis=-1)
    # if no valid actions (terminal), allow dummy action 0
    dummy = jax.nn.one_hot(jnp.zeros_like(has_any, dtype=jnp.int32), num_actions).astype(jnp.bool_)
    return jnp.where(has_any[:, None], flat, dummy)


def apply_action_mask(logits: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
    """Set invalid action logits to large negative value."""
    return jnp.where(valid, logits.astype(jnp.float32), NEG_INF)


def masked_log_softmax(logits: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
    """Log-softmax over masked logits."""
    return jax.nn.log_softmax(apply_action_mask(logits, valid), axis=-1)


def log_prob_from_logits(log_probs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    """Extract log-prob of chosen action. log_probs: (B, A), action: (B,)"""
    return jnp.take_along_axis(log_probs, action[:, None], axis=-1).squeeze(-1)


def entropy_from_log_probs(log_probs: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy from log-softmax. Invalid actions have ~0 prob so they're safe."""
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


# =============================================================================
# NN
# =============================================================================
# Same Transformer architecture as Expert Iteration for fair comparison.
# Cross-attention between EMS (empty spaces) and items is the key insight.
#
# Architecture:
#   EMS tokens  →  [self-attn] → [cross-attn with items] → policy head (E×I logits)
#   Item tokens →  [self-attn] → [cross-attn with EMS]   → value head (scalar)

class TransformerBlock(hk.Module):
    """Fallback Transformer block if Jumanji's isn't available."""

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        mlp_units: Sequence[int],
        w_init_scale: float,
        model_size: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_heads = int(num_heads)
        self.key_size = int(key_size)
        self.mlp_units = tuple(int(x) for x in mlp_units)
        self.model_size = int(model_size)
        self.w_init_scale = float(w_init_scale)

    def __call__(self, q, k, v, mask):
        ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln1")
        ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln2")

        q0, k0, v0 = ln1(q), ln1(k), ln1(v)

        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init=hk.initializers.VarianceScaling(self.w_init_scale),
            model_size=self.model_size,
            name="mha",
        )
        x = q + attn(q0, k0, v0, mask=mask)

        y = ln2(x)
        mlp = hk.nets.MLP((*self.mlp_units, self.model_size), name="mlp")
        return x + mlp(y)


class BinPackEncoder(hk.Module):
    """
    Transformer encoder for BinPack observations.
    
    Processes EMS and Item tokens through self-attention and cross-attention.
    The cross-attention is gated by action_mask to focus on valid placements.
    """

    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_layers = int(num_transformer_layers)
        self.num_heads = int(transformer_num_heads)
        self.key_size = int(transformer_key_size)
        self.mlp_units = tuple(int(x) for x in transformer_mlp_units)
        self.model_size = self.num_heads * self.key_size

    def __call__(self, observation) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # embed raw features
        ems_emb = self._embed_ems(observation.ems)
        items_emb = self._embed_items(observation.items)

        # attention masks
        ems_self_mask = self._make_self_attn_mask(observation.ems_mask)
        items_avail = observation.items_mask & ~observation.items_placed
        items_self_mask = self._make_self_attn_mask(items_avail)

        # cross-attention masks from action_mask
        ems_cross_items = jnp.expand_dims(observation.action_mask, axis=-3)
        items_cross_ems = jnp.expand_dims(jnp.moveaxis(observation.action_mask, -1, -2), axis=-3)

        # transformer layers
        for i in range(self.num_layers):
            w_scale = 2.0 / max(self.num_layers, 1)

            # self-attention
            ems_emb = TransformerBlock(
                self.num_heads, self.key_size, self.mlp_units, w_scale, self.model_size,
                name=f"ems_self_{i}"
            )(ems_emb, ems_emb, ems_emb, ems_self_mask)

            items_emb = TransformerBlock(
                self.num_heads, self.key_size, self.mlp_units, w_scale, self.model_size,
                name=f"items_self_{i}"
            )(items_emb, items_emb, items_emb, items_self_mask)

            # cross-attention (bidirectional)
            new_ems = TransformerBlock(
                self.num_heads, self.key_size, self.mlp_units, w_scale, self.model_size,
                name=f"ems_cross_{i}"
            )(ems_emb, items_emb, items_emb, ems_cross_items)

            items_emb = TransformerBlock(
                self.num_heads, self.key_size, self.mlp_units, w_scale, self.model_size,
                name=f"items_cross_{i}"
            )(items_emb, ems_emb, ems_emb, items_cross_ems)

            ems_emb = new_ems

        return ems_emb, items_emb

    def _embed_ems(self, ems) -> jnp.ndarray:
        features = jnp.stack(jax.tree_util.tree_leaves(ems), axis=-1).astype(jnp.float32)
        return hk.Linear(self.model_size, name="ems_proj")(features)

    def _embed_items(self, items) -> jnp.ndarray:
        features = jnp.stack(jax.tree_util.tree_leaves(items), axis=-1).astype(jnp.float32)
        return hk.Linear(self.model_size, name="items_proj")(features)

    @staticmethod
    def _make_self_attn_mask(mask_1d: jnp.ndarray) -> jnp.ndarray:
        mask = jnp.einsum("...i,...j->...ij", mask_1d, mask_1d)
        return jnp.expand_dims(mask, axis=-3)


class BinPackActorCritic(hk.Module):
    """
    Actor-Critic network for PPO.
    
    Outputs:
      - logits: (B, num_actions) for policy
      - value: (B,) in [0, 1] representing expected utilization
    """

    def __init__(self, name: str = "binpack_ac"):
        super().__init__(name=name)
        self.encoder = BinPackEncoder(
            num_transformer_layers=config.num_transformer_layers,
            transformer_num_heads=config.transformer_num_heads,
            transformer_key_size=config.transformer_key_size,
            transformer_mlp_units=config.transformer_mlp_units,
            name="encoder",
        )
        self.model_size = self.encoder.model_size

    def __call__(self, observation, is_training: bool = True):
        del is_training  # no dropout

        ems_emb, items_emb = self.encoder(observation)

        # policy head: bilinear over EMS × Items
        ems_h = hk.Linear(self.model_size, name="policy_ems")(ems_emb)
        items_h = hk.Linear(self.model_size, name="policy_items")(items_emb)
        logits_2d = jnp.einsum("...ek,...ik->...ei", ems_h, items_h)
        logits_2d = jnp.where(observation.action_mask, logits_2d, jnp.finfo(jnp.float32).min)
        logits = logits_2d.reshape(logits_2d.shape[0], -1)

        # value head: pooled embeddings → scalar
        ems_pooled = jnp.sum(ems_emb, axis=-2, where=observation.ems_mask[..., None])
        items_avail = observation.items_mask & ~observation.items_placed
        items_pooled = jnp.sum(items_emb, axis=-2, where=items_avail[..., None])

        combined = jnp.concatenate([ems_pooled, items_pooled], axis=-1)
        value = hk.nets.MLP([self.model_size, self.model_size, 1], name="value_head")(combined)
        value = jax.nn.sigmoid(jnp.squeeze(value, axis=-1))

        return logits, value


def forward_fn(observation, is_eval: bool = False):
    net = BinPackActorCritic()
    return net(observation, is_training=not is_eval)


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))


# =============================================================================
# Rollout data structures
# =============================================================================

class Transition(NamedTuple):
    """Single step of experience."""
    obs: object
    action: jnp.ndarray      # (B,)
    log_prob: jnp.ndarray    # (B,)
    value: jnp.ndarray       # (B,)
    reward: jnp.ndarray      # (B,)
    discount: jnp.ndarray    # (B,)
    mask: jnp.ndarray        # (B,) bool - active (not past terminal)
    entropy: jnp.ndarray     # (B,)


class Experience(NamedTuple):
    """Full episode experience with computed advantages."""
    obs: object
    actions: jnp.ndarray
    old_log_probs: jnp.ndarray
    old_values: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    mask: jnp.ndarray
    rewards: jnp.ndarray
    entropy: jnp.ndarray


# =============================================================================
# GAE
# =============================================================================
# GAE reduces variance in advantage estimates by exponentially weighting
# TD errors. Lambda=1 gives Monte Carlo, lambda=0 gives 1-step TD.
#
# A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
# where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
#
# See: https://arxiv.org/abs/1506.02438

def compute_gae(
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    values: jnp.ndarray,
    last_value: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute GAE advantages and returns.
    
    Args:
        rewards: (T, B)
        discounts: (T, B)
        values: (T, B)
        last_value: (B,) bootstrap value at t=T
    
    Returns:
        advantages: (T, B)
        returns: (T, B)
    """
    # V(t+1) for TD error computation
    values_next = jnp.concatenate([values[1:], last_value[None, :]], axis=0)
    deltas = rewards + discounts * values_next - values

    # reverse scan to accumulate advantages
    def scan_fn(gae, inputs):
        delta, discount = inputs
        gae = delta + discount * config.gae_lambda * gae
        return gae, gae

    _, advantages_rev = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(last_value),
        (deltas[::-1], discounts[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values

    return advantages, returns


# =============================================================================
# Exp collection
# =============================================================================
# Run episodes with current policy, record (obs, action, reward, ...) tuples.
# This is where PPO differs from Expert Iteration: we sample single actions
# from the policy, not full distributions from MCTS.

@jax.pmap
def collect_experience(model, rng_key: jnp.ndarray) -> Experience:
    """
    Collect rollouts for one iteration.
    
    Runs rollout_batch_size / num_devices episodes on each device.
    """
    params, net_state = model
    batch_per_device = config.rollout_batch_size // num_devices

    rng_key, reset_key, scan_key = jax.random.split(rng_key, 3)
    reset_keys = jax.random.split(reset_key, batch_per_device)

    state, timestep = jax.vmap(env.reset)(reset_keys)
    done = jnp.zeros((batch_per_device,), dtype=jnp.bool_)

    def step_fn(carry, step_key):
        state, timestep, done = carry
        obs = timestep.observation

        # forward pass
        (logits, value), _ = forward.apply(params, net_state, obs, is_eval=True)

        # sample action from policy
        valid_mask = get_valid_action_mask(obs.action_mask)
        log_probs_all = masked_log_softmax(logits, valid_mask)
        entropy = entropy_from_log_probs(log_probs_all)

        action = jax.random.categorical(
            step_key, apply_action_mask(logits, valid_mask), axis=-1
        ).astype(jnp.int32)
        log_prob = log_prob_from_logits(log_probs_all, action)

        # step environment
        action_pair = unflatten_action(action)
        next_state, next_timestep = jax.vmap(env.step)(state, action_pair)

        # mask out data after episode termination
        active = ~done
        reward = jnp.where(active, next_timestep.reward.astype(jnp.float32), 0.0)
        discount = jnp.where(active, next_timestep.discount.astype(jnp.float32) * config.discount, 0.0)

        next_done = done | (next_timestep.discount == 0.0)

        # zero out signals for terminated episodes
        transition = Transition(
            obs=obs,
            action=jnp.where(active, action, 0),
            log_prob=jnp.where(active, log_prob.astype(jnp.float32), 0.0),
            value=jnp.where(active, value.astype(jnp.float32), 0.0),
            reward=reward,
            discount=discount,
            mask=active,
            entropy=jnp.where(active, entropy.astype(jnp.float32), 0.0),
        )

        return (next_state, next_timestep, next_done), transition

    step_keys = jax.random.split(scan_key, config.max_num_steps)
    (final_state, final_timestep, final_done), trajectory = jax.lax.scan(
        step_fn, (state, timestep, done), step_keys
    )

    # bootstrap value for GAE
    (_, final_value), _ = forward.apply(params, net_state, final_timestep.observation, is_eval=True)
    last_value = jnp.where(final_done, 0.0, final_value.astype(jnp.float32))

    advantages, returns = compute_gae(
        trajectory.reward, trajectory.discount, trajectory.value, last_value
    )

    return Experience(
        obs=trajectory.obs,
        actions=trajectory.action,
        old_log_probs=trajectory.log_prob,
        old_values=trajectory.value,
        advantages=advantages,
        returns=returns,
        mask=trajectory.mask,
        rewards=trajectory.reward,
        entropy=trajectory.entropy,
    )


# =============================================================================
# PPO Loss
# =============================================================================
# The PPO objective has three parts:
#   1. Clipped policy loss: encourages improvement while preventing large changes
#   2. Value loss: train critic to predict returns
#   3. Entropy bonus: encourages exploration
# The clipping is the key innovation: it creates a "trust region" that
# prevents the policy from changing too much in one update.
# My favorite lecture on this is https://www.youtube.com/watch?v=ySenCHPsKJU&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=36
# Generally the playlist is great ^
class Batch(NamedTuple):
    """Training batch."""
    obs: object
    actions: jnp.ndarray
    old_log_probs: jnp.ndarray
    old_values: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    mask: jnp.ndarray


def ppo_loss(params, net_state, batch: Batch):
    """
    Compute PPO loss with clipped objective.
    
    Returns:
        total_loss: scalar
        aux: (new_net_state, policy_loss, value_loss, entropy, approx_kl, clip_fraction)
    """
    (logits, value), net_state = forward.apply(params, net_state, batch.obs, is_eval=False)

    valid_mask = get_valid_action_mask(batch.obs.action_mask)
    log_probs_all = masked_log_softmax(logits, valid_mask)
    new_log_prob = log_prob_from_logits(log_probs_all, batch.actions).astype(jnp.float32)
    entropy = entropy_from_log_probs(log_probs_all).astype(jnp.float32)

    mask_f = batch.mask.astype(jnp.float32)
    num_valid = jnp.maximum(jnp.sum(mask_f), 1.0)

    # policy loss (clipped surrogate)
    # ratio = pi_new(a|s) / pi_old(a|s)
    log_ratio = new_log_prob - batch.old_log_probs
    ratio = jnp.exp(log_ratio)

    # clipped objective: min(ratio * A, clip(ratio) * A)
    surr1 = ratio * batch.advantages
    surr2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * batch.advantages
    policy_loss = -jnp.sum(jnp.minimum(surr1, surr2) * mask_f) / num_valid

    # value loss (optionally clipped)
    v = value.astype(jnp.float32)
    v_old = batch.old_values
    v_clipped = v_old + jnp.clip(v - v_old, -config.value_clip_eps, config.value_clip_eps)
    v_loss1 = (v - batch.returns) ** 2
    v_loss2 = (v_clipped - batch.returns) ** 2
    value_loss = 0.5 * jnp.sum(jnp.maximum(v_loss1, v_loss2) * mask_f) / num_valid

    # entropy bonus (encourages exploration)
    mean_entropy = jnp.sum(entropy * mask_f) / num_valid

    # total loss
    total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * mean_entropy

    # diagnostics
    approx_kl = jnp.sum((batch.old_log_probs - new_log_prob) * mask_f) / num_valid
    clip_fraction = jnp.sum(
        (jnp.abs(ratio - 1.0) > config.clip_eps).astype(jnp.float32) * mask_f
    ) / num_valid

    return total_loss, (net_state, policy_loss, value_loss, mean_entropy, approx_kl, clip_fraction)


# =============================================================================
# Optimizer and train step
# =============================================================================

def make_optimizer():
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )


optimizer = make_optimizer()


@partial(jax.pmap, axis_name="devices")
def train_step(model, opt_state, batch: Batch):
    """Single PPO update with gradient sync across devices."""
    params, net_state = model

    (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, net_state, batch)
    new_state, policy_loss, value_loss, entropy, approx_kl, clip_frac = aux

    # sync gradients across devices
    grads = jax.lax.pmean(grads, axis_name="devices")

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # sync metrics for logging
    loss = jax.lax.pmean(loss, axis_name="devices")
    policy_loss = jax.lax.pmean(policy_loss, axis_name="devices")
    value_loss = jax.lax.pmean(value_loss, axis_name="devices")
    entropy = jax.lax.pmean(entropy, axis_name="devices")
    approx_kl = jax.lax.pmean(approx_kl, axis_name="devices")
    clip_frac = jax.lax.pmean(clip_frac, axis_name="devices")

    return (params, new_state), opt_state, loss, policy_loss, value_loss, entropy, approx_kl, clip_frac


# =============================================================================
# Eval
# =============================================================================

@jax.pmap
def evaluate_greedy(model, rng_key: jnp.ndarray):
    """Evaluate with greedy action selection (argmax policy)."""
    params, net_state = model
    batch_per_device = config.eval_batch_size // num_devices

    reset_keys = jax.random.split(rng_key, batch_per_device)
    state, timestep = jax.vmap(env.reset)(reset_keys)

    done = jnp.zeros((batch_per_device,), dtype=jnp.bool_)
    total_return = jnp.zeros((batch_per_device,), dtype=jnp.float32)
    episode_length = jnp.zeros((batch_per_device,), dtype=jnp.int32)

    def step_fn(carry, _):
        state, timestep, done, total_return, episode_length = carry
        obs = timestep.observation

        (logits, _), _ = forward.apply(params, net_state, obs, is_eval=True)
        valid_mask = get_valid_action_mask(obs.action_mask)
        action = jnp.argmax(apply_action_mask(logits, valid_mask), axis=-1).astype(jnp.int32)

        action_pair = unflatten_action(action)
        next_state, next_timestep = jax.vmap(env.step)(state, action_pair)

        active = ~done
        reward = jnp.where(active, next_timestep.reward.astype(jnp.float32), 0.0)
        total_return = total_return + reward
        episode_length = episode_length + active.astype(jnp.int32)
        done = done | (next_timestep.discount == 0.0)

        return (next_state, next_timestep, done, total_return, episode_length), None

    (_, _, _, total_return, episode_length), _ = jax.lax.scan(
        step_fn,
        (state, timestep, done, total_return, episode_length),
        None,
        length=config.max_num_steps,
    )

    return total_return, episode_length


# Main training loop
if __name__ == "__main__":
    os.makedirs(config.save_dir, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.run_name_prefix}_{config.env_id}_seed{config.seed}_{timestamp}"

    wandb.init(
        project=config.wandb_project,
        config=config.model_dump(),
        name=run_name,
        dir=config.save_dir,
    )

    # rng
    rng_key = jax.random.PRNGKey(config.seed)

    # init network
    rng_key, env_key, init_key = jax.random.split(rng_key, 3)
    dummy_state, dummy_timestep = env.reset(env_key)
    dummy_obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), dummy_timestep.observation)

    params, net_state = forward.init(init_key, dummy_obs)
    model = (params, net_state)
    opt_state = optimizer.init(params)

    # replicate across devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # checkpoint dir
    ckpt_dir = os.path.join(config.save_dir, config.env_id, "ppo", f"seed_{config.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # stats
    iteration = 0
    total_frames = 0.0
    total_hours = 0.0

    print("=" * 60)
    print("Starting PPO Training")
    print("=" * 60)

    while True:
        # eval
        if iteration % config.eval_interval == 0:
            rng_key, eval_key = jax.random.split(rng_key)
            eval_keys = jax.random.split(eval_key, num_devices)
            returns, lengths = evaluate_greedy(model, eval_keys)

            avg_return = float(returns.mean())
            avg_length = float(lengths.mean())

            wandb.log({
                "iteration": iteration,
                "eval/greedy/avg_return": avg_return,
                "eval/greedy/avg_steps": avg_length,
                "hours": total_hours,
                "frames": total_frames,
            })
            print(f"[eval] iter={iteration:4d} | return={avg_return:.4f} | steps={avg_length:.1f}")

        if iteration % config.save_interval == 0:
            model_single = jax.tree_util.tree_map(lambda x: x[0], model)
            opt_state_single = jax.tree_util.tree_map(lambda x: x[0], opt_state)

            ckpt_path = os.path.join(ckpt_dir, f"iter_{iteration:06d}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "config": config.model_dump(),
                    "rng_key": jax.device_get(rng_key),
                    "model": jax.device_get(model_single),
                    "opt_state": jax.device_get(opt_state_single),
                    "iteration": iteration,
                    "frames": total_frames,
                    "hours": total_hours,
                    "env_id": config.env_id,
                    "jumanji_version": getattr(jumanji, "__version__", "unknown"),                }, f)
            print(f"[ckpt] Saved: {ckpt_path}")

        if iteration >= config.max_num_iters:
            print("=" * 60)
            print("Training Complete!")
            print("=" * 60)
            break

        iter_start = time.time()

        # collect experience
        rng_key, collect_key = jax.random.split(rng_key)
        collect_keys = jax.random.split(collect_key, num_devices)

        experience = collect_experience(model, collect_keys)

        # stats from rollout
        exp_host = jax.device_get(experience)
        episode_returns = exp_host.rewards.sum(axis=1)  # sum over time
        episode_lengths = exp_host.mask.sum(axis=1)
        avg_rollout_return = float(episode_returns.mean())
        avg_rollout_length = float(episode_lengths.mean())

        total_frames += float(exp_host.mask.sum())

        # prepare training batches
        def flatten_samples(x):
            return x.reshape((-1,) + x.shape[3:])

        flat = jax.tree_util.tree_map(flatten_samples, exp_host)

        # normalize advantages (important for PPO stability!)
        adv = flat.advantages
        mask_f = flat.mask.astype(jnp.float32)
        num_valid = float(jnp.maximum(mask_f.sum(), 1.0))
        adv_mean = float((adv * mask_f).sum() / num_valid)
        adv_var = float(((adv - adv_mean) ** 2 * mask_f).sum() / num_valid)
        adv_std = (adv_var + 1e-8) ** 0.5
        adv_normalized = ((adv - adv_mean) / adv_std).astype(jnp.float32)

        batch_all = Batch(
            obs=flat.obs,
            actions=flat.actions,
            old_log_probs=flat.old_log_probs,
            old_values=flat.old_values,
            advantages=adv_normalized,
            returns=flat.returns.astype(jnp.float32),
            mask=flat.mask,
        )

        N = batch_all.actions.shape[0]
        num_updates = N // config.training_batch_size
        N_used = num_updates * config.training_batch_size

        if num_updates == 0:
            print(f"[warn] Not enough samples: {N} < {config.training_batch_size}")
            iteration += 1
            continue

        per_device = config.training_batch_size // num_devices

        # ppo updates (multiple epochs over the data)
        losses, pol_losses, val_losses = [], [], []
        entropies, kls, clip_fracs = [], [], []

        for epoch in range(config.num_epochs):
            rng_key, shuffle_key = jax.random.split(rng_key)
            perm = jax.device_get(jax.random.permutation(shuffle_key, jnp.arange(N_used)))

            batch_shuffled = jax.tree_util.tree_map(lambda x: x[:N_used][perm], batch_all)

            def to_minibatches(x):
                return x.reshape((num_updates, num_devices, per_device) + x.shape[1:])

            minibatches = jax.tree_util.tree_map(to_minibatches, batch_shuffled)

            for i in range(num_updates):
                mb = jax.tree_util.tree_map(lambda x: x[i], minibatches)
                model, opt_state, loss, pl, vl, ent, kl, cf = train_step(model, opt_state, mb)

                losses.append(float(jax.device_get(loss[0])))
                pol_losses.append(float(jax.device_get(pl[0])))
                val_losses.append(float(jax.device_get(vl[0])))
                entropies.append(float(jax.device_get(ent[0])))
                kls.append(float(jax.device_get(kl[0])))
                clip_fracs.append(float(jax.device_get(cf[0])))

        iter_time = time.time() - iter_start
        total_hours += iter_time / 3600.0


        metrics = {
            "iteration": iteration,
            "hours": total_hours,
            "frames": total_frames,
            "rollout/avg_return": avg_rollout_return,
            "rollout/avg_steps": avg_rollout_length,
            "train/loss": sum(losses) / len(losses),
            "train/policy_loss": sum(pol_losses) / len(pol_losses),
            "train/value_loss": sum(val_losses) / len(val_losses),
            "train/entropy": sum(entropies) / len(entropies),
            "train/approx_kl": sum(kls) / len(kls),
            "train/clip_fraction": sum(clip_fracs) / len(clip_fracs),
            "adv/mean": adv_mean,
            "adv/std": adv_std,
        }
        wandb.log(metrics)

        print(f"[train] iter={iteration:4d} | "
              f"return={avg_rollout_return:.4f} | "
              f"policy_loss={sum(pol_losses)/len(pol_losses):.4f} | "
              f"kl={sum(kls)/len(kls):.4f} | "
              f"time={iter_time:.1f}s")

        iteration += 1

    wandb.finish()