#!/usr/bin/env python3
"""
================================================================================
Policy Iteration via Search Distillation for 3D Bin Packing
================================================================================

A clean implementation of search-based policy iteration for Jumanji's
JAX 3D BinPack environment. We use Monte Carlo Tree Search (MCTS) as a
policy improvement operator, then distill the improved policy into a
neural network via supervised learning.

┌─────────────────────────────────────────────────────────────────────────────┐
│  MCTS as a Policy Improvement Operator                                      │
│                                                                             │
│  Unlike board games (Go, Chess), 3D bin packing is a single-agent planning  │
│  task with known dynamics. We use Gumbel MuZero search to compute improved  │
│  policy targets—softmax(prior + Q)—then distill them into a neural network  │
│  via cross-entropy. This is policy iteration: search is the improvement     │
│  operator, supervised learning is how we internalize the improvement.       │
│                                                                             │
│  Related work: "Thinking Fast and Slow with Deep Learning and Tree Search"  │
│  (Expert Iteration) https://arxiv.org/abs/1705.08439                        │
│                                                                             │
│  Key difference: We use Gumbel MuZero's policy improvement operator         │
│  softmax(prior + Q) rather than visit count proportions n(s,a)/n(s).        │
│  See: https://openreview.net/forum?id=bERaNdoegnO                           │
└─────────────────────────────────────────────────────────────────────────────┘

Why this approach beats PPO for bin packing:
  1. Search provides dense supervision (full action distributions, not samples)
  2. We have exact environment dynamics (no need to learn a world model)
  3. Policy improvement is guided by lookahead search, not just gradient signals
  4. Reward influences policy indirectly through search, giving stable learning

JAX parallelization:
  - jax.vmap: Vectorize over batch dimension (parallel episodes)
  - jax.pmap: Distribute across multiple devices (GPUs/TPUs)
  Together: Massive throughput with minimal code changes

References and Acknowledgments:
  - Jumanji BinPack environment:
    https://github.com/instadeepai/jumanji/blob/main/jumanji/environments/packing/bin_pack/env.py
  - Jumanji A2C networks (transformer architecture):
    https://github.com/instadeepai/jumanji/blob/main/jumanji/training/networks/bin_pack/actor_critic.py
  - PGX AlphaZero implementation (JAX MCTS patterns):
    https://github.com/sotetsuk/pgx/tree/main/examples/alphazero
  - Expert Iteration paper:
    https://arxiv.org/abs/1705.08439
  - Gumbel MuZero paper (our policy improvement operator):
    https://openreview.net/forum?id=bERaNdoegnO
  - mctx library (DeepMind):
    https://github.com/google-deepmind/mctx

Requirements:
  pip install jax haiku optax mctx wandb omegaconf pydantic jumanji

Usage:
  python train_expert_iteration_binpack.py
  python train_expert_iteration_binpack.py --num_simulations 64 seed=123
  python train_expert_iteration_binpack.py env_id=BinPack-v2 training_batch_size=8192
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
import mctx
import optax
import wandb
from omegaconf import OmegaConf
from pydantic import BaseModel


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                              Config                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# I usually use Pydantic for type-safe configuration with validation.
# OmegaConf enables easy CLI overrides like: `python train.py learning_rate=1e-4`
#


class Config(BaseModel):
    """Training configuration with sensible defaults for BinPack-v2."""

    # Env
    env_id: str = "BinPack-v2"
    seed: int = 0

    # Training Loop
    max_num_iters: int = 800          # Total policy iteration cycles
    eval_interval: int = 20           # Evaluate every N iterations
    save_interval: int = 200          # Checkpoint every N iterations
    save_dir: str = "./checkpoints"

    # Search / Data Collection
    # These control the search that generates training targets.
    # More simulations = stronger policy improvement = better targets (but slower)

    rollout_batch_size: int = 1024    # Episodes per iteration (across all devices)
    num_simulations: int = 32         # MCTS simulations per decision
    max_num_steps: int = 20           # Max steps per episode (BinPack-v2 has ≤20 items)
    discount: float = 1.0             # Undiscounted for episodic tasks

    # NN HP
    training_batch_size: int = 4096   # Samples per SGD step (across all devices)
    learning_rate: float = 1e-3
    value_loss_weight: float = 1.0    # Balance policy vs value loss
    max_grad_norm: float = 1.0        # Gradient clipping for stability

    # Transformer-based architecture inspired by Jumanji's A2C networks.
    # Cross-attention between EMS (empty spaces) and items is key!
    num_transformer_layers: int = 4
    transformer_num_heads: int = 4
    transformer_key_size: int = 32
    transformer_mlp_units: Sequence[int] = (256, 256)

    # Eval
    eval_batch_size: int = 1024
    eval_use_mcts: bool = False       # Greedy policy eval (faster) vs MCTS eval
    eval_num_simulations: int = 32

    wandb_project: str = "jumanji-binpack"

    class Config:
        extra = "forbid"


# CLI Parsing

parser = argparse.ArgumentParser(
    description="Policy Iteration via Search Distillation for 3D Bin Packing",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--num_simulations", type=int, default=8)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--eval_num_simulations", type=int, default=None)
args, unknown = parser.parse_known_args()

# Merge CLI overrides with defaults
conf_dict = OmegaConf.to_container(OmegaConf.from_cli(unknown), resolve=True) or {}
if args.num_simulations is not None:
    conf_dict["num_simulations"] = args.num_simulations
if args.seed is not None:
    conf_dict["seed"] = args.seed
if args.eval_num_simulations is not None:
    conf_dict["eval_num_simulations"] = args.eval_num_simulations

config = Config(**conf_dict)
print("=" * 60)
print("Configuration:")
print("=" * 60)
print(config)
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           Jax Device Setup                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# JAX automatically detects available devices (CPUs, GPUs, TPUs).
# We use jax.pmap to distribute computation across all devices.
#
#   - Each device processes a shard of the batch independently
#   - Gradients are synchronized with jax.lax.pmean across devices
#   - Batch sizes must be divisible by num_devices
#
# See: https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html

devices = jax.local_devices()
num_devices = len(devices)
print(f"JAX devices: {devices}")
print(f"Number of devices: {num_devices}")

assert config.rollout_batch_size % num_devices == 0, \
    f"rollout_batch_size ({config.rollout_batch_size}) must be divisible by num_devices ({num_devices})"
assert config.training_batch_size % num_devices == 0, \
    f"training_batch_size ({config.training_batch_size}) must be divisible by num_devices ({num_devices})"
assert config.eval_batch_size % num_devices == 0, \
    f"eval_batch_size ({config.eval_batch_size}) must be divisible by num_devices ({num_devices})"
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           Env Setup                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Jumanji's BinPack environment:
#   - State: Container with Empty Maximal Spaces (EMS) + items to pack
#   - Action: (ems_id, item_id) - place item at the corner of an EMS
#   - Reward (Dense): Volume utilization gained by placing an item
#
# The action space is MultiDiscrete([obs_num_ems, max_num_items]).
# We flatten this to a single discrete action for simpler MCTS handling.
#
# See: https://github.com/instadeepai/jumanji/blob/main/jumanji/environments/packing/bin_pack/

import jumanji

env = jumanji.make(config.env_id)

# Extract action space dimensions
obs_num_ems = int(getattr(env, "obs_num_ems", env.action_spec.num_values[0]))
max_num_items = int(
    getattr(getattr(env, "generator", None), "max_num_items", env.action_spec.num_values[1])
)
num_actions = obs_num_ems * max_num_items

print(f"Environment: {config.env_id}")
print(f"  obs_num_ems (observable empty spaces): {obs_num_ems}")
print(f"  max_num_items: {max_num_items}")
print(f"  Flattened action space size: {num_actions}")
print()



# Helper functions to convert between flat action indices and (ems_id, item_id) pairs.

def unflatten_action(action: jnp.ndarray) -> jnp.ndarray:
    """
    Convert flat action index to (ems_id, item_id) pair.

    Args:
        action: Flat action indices, shape (batch_size,)

    Returns:
        Action pairs, shape (batch_size, 2) as int32

    Example:
        If max_num_items=20, action=45 → ems_id=2, item_id=5
    """
    ems_id = action // max_num_items
    item_id = action % max_num_items
    return jnp.stack([ems_id, item_id], axis=-1).astype(jnp.int32)


def get_valid_action_mask(action_mask_2d: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten 2D action mask and ensure at least one action is valid.

    The environment's action_mask has shape (batch, obs_num_ems, max_num_items).
    MCTS requires at least one valid action per state, so we add a dummy
    action if the episode has terminated.

    Args:
        action_mask_2d: Boolean mask, shape (batch, obs_num_ems, max_num_items)

    Returns:
        Flattened mask, shape (batch, num_actions) with ≥1 True per row
    """
    flat = action_mask_2d.reshape((action_mask_2d.shape[0], -1))
    has_valid_action = jnp.any(flat, axis=-1)  # (batch,)

    # Create dummy mask allowing action 0 (used for terminated states)
    dummy = jax.nn.one_hot(
        jnp.zeros_like(has_valid_action, dtype=jnp.int32), num_actions
    ).astype(jnp.bool_)

    return jnp.where(has_valid_action[:, None], flat, dummy)


def apply_action_mask(logits: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Mask invalid actions by setting their logits to a large negative value.

    Idea: We use a finite minimum (not -inf) to avoid NaN issues in softmax.
    We also center logits first for numerical stability.

    Args:
        logits: Raw policy logits, shape (..., num_actions)
        valid_mask: Boolean mask, shape (..., num_actions)

    Returns:
        Masked logits with invalid actions set to ~-3.4e38
    """
    # Center logits for numerical stability
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(valid_mask, logits, jnp.finfo(logits.dtype).min)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          NN                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# The cross-attention is KEY: it lets the network reason about
# which items fit in which spaces. The action mask gates this attention
# to only consider valid (EMS, item) pairs.
#
# See Jumanji's A2C networks for the original inspiration:
# https://github.com/instadeepai/jumanji/blob/main/jumanji/training/networks/bin_pack/actor_critic.py


class TransformerBlock(hk.Module):
    """
    Minimal Transformer block using Haiku's MultiHeadAttention.
    """

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
        self.num_heads = num_heads
        self.key_size = key_size
        self.mlp_units = tuple(mlp_units)
        self.model_size = model_size
        self.w_init_scale = w_init_scale

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Args:
            query, key, value: Input tensors, shape (..., seq_len, model_size)
            mask: Attention mask, shape (..., 1, query_len, key_len)

        Returns:
            Output tensor, same shape as query
        """
        # Pre-norm (LayerNorm before attention)
        ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln1")
        ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln2")

        q_norm = ln1(query)
        k_norm = ln1(key)
        v_norm = ln1(value)

        # Multi-head attention
        attn = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            w_init=hk.initializers.VarianceScaling(self.w_init_scale),
            model_size=self.model_size,
            name="mha",
        )
        attn_out = attn(q_norm, k_norm, v_norm, mask=mask)

        # First residual connection
        x = query + attn_out

        # FFN with second residual
        y = ln2(x)
        mlp = hk.nets.MLP((*self.mlp_units, self.model_size), name="mlp")
        return x + mlp(y)


class BinPackEncoder(hk.Module):
    """
    Transformer encoder for BinPack observations.

    Processes EMS (Empty Maximal Space) and Item tokens through:
    1. Self-attention within each token type
    2. Cross-attention between EMS and Items (bidirectional)

    The cross-attention is gated by the action_mask, ensuring the network
    only attends to valid (EMS, item) placement combinations.
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
        """
        Encode observation into EMS and Item embeddings.

        Args:
            observation: Jumanji BinPack observation containing:
                - ems: EMS coordinates (x1,x2,y1,y2,z1,z2)
                - ems_mask: Which EMSs are valid
                - items: Item dimensions (x_len, y_len, z_len)
                - items_mask: Which items exist
                - items_placed: Which items are already placed
                - action_mask: Valid (ems, item) pairs

        Returns:
            (ems_embeddings, items_embeddings): Both shape (..., seq_len, model_size)
        """
        
        # Embed raw features into model dimension
        ems_embeddings = self._embed_ems(observation.ems)
        items_embeddings = self._embed_items(observation.items)

        # Build attention masks
        # Self-attention: tokens can only attend to valid tokens of same type
        # Cross-attention: EMS <-> Items gated by action_mask
        ems_self_mask = self._make_self_attention_mask(observation.ems_mask)

        # Items mask: only attend to available (valid & not yet placed) items
        items_available = observation.items_mask & ~observation.items_placed
        items_self_mask = self._make_self_attention_mask(items_available)

        # Cross-attention masks (action_mask gates which EMS-item pairs interact)
        # Shape: (..., 1, query_len, key_len)
        ems_cross_items_mask = jnp.expand_dims(observation.action_mask, axis=-3)
        items_cross_ems_mask = jnp.expand_dims(
            jnp.moveaxis(observation.action_mask, -1, -2), axis=-3
        )

        for layer_idx in range(self.num_layers):
            # Scale initialization based on depth (helps with deep networks)
            w_init_scale = 2.0 / max(self.num_layers, 1)

            # EMS self-attention
            ems_embeddings = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                mlp_units=self.mlp_units,
                w_init_scale=w_init_scale,
                model_size=self.model_size,
                name=f"ems_self_attn_{layer_idx}",
            )(ems_embeddings, ems_embeddings, ems_embeddings, ems_self_mask)

            # Items self-attention
            items_embeddings = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                mlp_units=self.mlp_units,
                w_init_scale=w_init_scale,
                model_size=self.model_size,
                name=f"items_self_attn_{layer_idx}",
            )(items_embeddings, items_embeddings, items_embeddings, items_self_mask)

            # Bidirectional cross-attention
            new_ems = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                mlp_units=self.mlp_units,
                w_init_scale=w_init_scale,
                model_size=self.model_size,
                name=f"ems_cross_items_{layer_idx}",
            )(ems_embeddings, items_embeddings, items_embeddings, ems_cross_items_mask)

            items_embeddings = TransformerBlock(
                num_heads=self.num_heads,
                key_size=self.key_size,
                mlp_units=self.mlp_units,
                w_init_scale=w_init_scale,
                model_size=self.model_size,
                name=f"items_cross_ems_{layer_idx}",
            )(items_embeddings, ems_embeddings, ems_embeddings, items_cross_ems_mask)

            ems_embeddings = new_ems

        return ems_embeddings, items_embeddings

    def _embed_ems(self, ems) -> jnp.ndarray:
        """Embed EMS coordinates: (x1,x2,y1,y2,z1,z2) → model_size"""
        ems_features = jnp.stack(jax.tree_util.tree_leaves(ems), axis=-1).astype(jnp.float32)
        return hk.Linear(self.model_size, name="ems_projection")(ems_features)

    def _embed_items(self, items) -> jnp.ndarray:
        """Embed item dimensions: (x_len, y_len, z_len) → model_size"""
        item_features = jnp.stack(jax.tree_util.tree_leaves(items), axis=-1).astype(jnp.float32)
        return hk.Linear(self.model_size, name="item_projection")(item_features)

    @staticmethod
    def _make_self_attention_mask(token_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Create self-attention mask from token validity mask.

        Args:
            token_mask: Boolean mask, shape (..., seq_len)

        Returns:
            Attention mask, shape (..., 1, seq_len, seq_len)
        """
        # Outer product: token i can attend to token j if both are valid
        mask = jnp.einsum("...i,...j->...ij", token_mask, token_mask)
        return jnp.expand_dims(mask, axis=-3)


class BinPackPolicyValueNet(hk.Module):
    """
    Combined policy and value network for search distillation.

    Outputs:
        - Policy logits: (batch, obs_num_ems * max_num_items)
        - Value: (batch,) in [0, 1] (bin utilization)
    """

    def __init__(
        self,
        num_transformer_layers: int,
        transformer_num_heads: int,
        transformer_key_size: int,
        transformer_mlp_units: Sequence[int],
        name: str = "binpack_policy_value_net",
    ):
        super().__init__(name=name)
        self.encoder = BinPackEncoder(
            num_transformer_layers=num_transformer_layers,
            transformer_num_heads=transformer_num_heads,
            transformer_key_size=transformer_key_size,
            transformer_mlp_units=transformer_mlp_units,
            name="encoder",
        )
        self.model_size = self.encoder.model_size

    def __call__(self, observation, is_training: bool = True):
        del is_training

        ems_embeddings, items_embeddings = self.encoder(observation)

        # Policy Head
        # Compute logits via bilinear form: ems_h @ items_h^T
        # This naturally produces (obs_num_ems, max_num_items) shaped logits
        ems_policy = hk.Linear(self.model_size, name="policy_ems")(ems_embeddings)
        items_policy = hk.Linear(self.model_size, name="policy_items")(items_embeddings)

        # Bilinear: logits[e,i] = ems_policy[e] · items_policy[i]
        logits_2d = jnp.einsum("...ek,...ik->...ei", ems_policy, items_policy)

        # Apply action mask
        logits_2d = jnp.where(
            observation.action_mask,
            logits_2d,
            jnp.finfo(jnp.float32).min
        )

        # Flatten to single discrete action space
        logits = logits_2d.reshape(*logits_2d.shape[:-2], -1)

        # Value Head
        # Pool embeddings and predict utilization ∈ [0, 1]
        # Sum-pool valid EMS embeddings
        ems_pooled = jnp.sum(
            ems_embeddings,
            axis=-2,
            where=observation.ems_mask[..., None]
        )

        # Sum-pool available item embeddings
        items_available = observation.items_mask & ~observation.items_placed
        items_pooled = jnp.sum(
            items_embeddings,
            axis=-2,
            where=items_available[..., None]
        )

        # Concatenate and predict value
        combined = jnp.concatenate([ems_pooled, items_pooled], axis=-1)
        value = hk.nets.MLP(
            [self.model_size, self.model_size, 1],
            name="value_head"
        )(combined)
        value = jnp.squeeze(value, axis=-1)

        # Sigmoid: value represents expected utilization ∈ [0, 1]
        value = jax.nn.sigmoid(value)

        return logits, value


# Haiku Function Transform
# Haiku uses functional transforms to convert stateful modules into pure functions.
# `hk.transform_with_state` handles both parameters and state (e.g., BatchNorm stats).
# See: https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html
# ─────────────────────────────────────────────────────────────────────────────

def forward_fn(observation, is_eval: bool = False):
    """Network forward pass (Haiku-style)."""
    net = BinPackPolicyValueNet(
        num_transformer_layers=config.num_transformer_layers,
        transformer_num_heads=config.transformer_num_heads,
        transformer_key_size=config.transformer_key_size,
        transformer_mlp_units=config.transformer_mlp_units,
    )
    return net(observation, is_training=not is_eval)


# Remove RNG dependency from apply (we don't use dropout)
forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       MCTS Recurrent Func                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# The mctx library requires a "recurrent function" that:
#   1. Takes current state + action
#   2. Returns next state + (reward, discount, prior_logits, value)
#
# For BinPack, we have PERFECT knowledge of the environment dynamics,
# so MCTS is doing planning with the true model (no learned dynamics).
#
# This is a huge advantage over model-based RL where dynamics are learned!
# See: https://github.com/google-deepmind/mctx


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state):
    """
    Environment model for MCTS simulation.

    This function is called during MCTS tree expansion to simulate
    what happens when taking an action from a state.

    Args:
        model: Tuple of (params, state) for the neural network
        rng_key: Random key (unused since env is deterministic)
        action: Flat action indices, shape (batch,)
        state: Current environment states

    Returns:
        output: mctx.RecurrentFnOutput with reward, discount, prior, value
        next_state: New environment states after taking action
    """
    del rng_key
    model_params, model_state = model

    # Convert flat action to (ems_id, item_id) and step environment
    action_pair = unflatten_action(action)
    next_state, timestep = jax.vmap(env.step)(state, action_pair)

    # Extract reward and discount
    reward = timestep.reward.astype(jnp.float32)
    discount = timestep.discount.astype(jnp.float32) * config.discount

    # Get network predictions for the next state
    observation = timestep.observation
    (logits, value), _ = forward.apply(model_params, model_state, observation, is_eval=True)

    # Mask invalid actions
    valid_mask = get_valid_action_mask(observation.action_mask)
    logits = apply_action_mask(logits, valid_mask)

    # Terminal states have zero value and discount
    is_terminal = timestep.discount == 0.0
    value = jnp.where(is_terminal, 0.0, value)
    discount = jnp.where(is_terminal, 0.0, discount)

    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    ), next_state


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    Policy Iteration via Search Distillation               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# This is the core of search distillation:
#
#   1. Run episodes using Gumbel MuZero search at each decision point
#   2. Record the search action distribution (not just the chosen action!)
#   3. Use these distributions as training targets for the policy network
#
# The search "action_weights" represent a much stronger policy than the raw
# network output—computed as softmax(prior + Q), a direct policy improvement
# operator. By training to match these, we iteratively improve the policy.
#
# Unlike policy gradient methods (PPO, A2C), we train on the
# full action distribution from search, not just sampled actions. This
# provides much richer learning signal and more stable training!
#
# Key insight: reward doesn't appear in the policy loss. It only influences
# the policy indirectly through how search evaluates actions.


class RolloutData(NamedTuple):
    """Data collected from a single rollout step."""
    observation: object          # Observation pytree
    action_weights: jnp.ndarray  # Search policy distribution (training target!)
    reward: jnp.ndarray          # Step reward
    discount: jnp.ndarray        # Discount factor (0 at terminal)
    mask: jnp.ndarray            # True for valid decision steps


@jax.pmap
def collect_experience(model, rng_key: jnp.ndarray) -> Tuple[RolloutData, object, jnp.ndarray]:
    """
    Collect training data by running search-guided episodes.

    This function:
    1. Resets a batch of environments
    2. At each step, runs Gumbel MuZero search to get an improved policy
    3. Samples actions from search distribution
    4. Records (observation, search_policy, reward, discount) tuples

    Args:
        model: Replicated (params, state) tuple
        rng_key: Random key for this device

    Returns:
        data: RolloutData with shape (max_steps, batch_per_device, ...)
        final_timestep: Terminal timesteps (for debugging)
        final_done: Terminal flags
    """
    model_params, model_state = model
    batch_per_device = config.rollout_batch_size // num_devices

    # Reset environments
    rng_key, reset_key, scan_key = jax.random.split(rng_key, 3)
    reset_keys = jax.random.split(reset_key, batch_per_device)

    # jax.vmap: Vectorize env.reset across the batch dimension
    # This processes all environments in parallel on a single device
    state, timestep = jax.vmap(env.reset)(reset_keys)
    done = jnp.zeros((batch_per_device,), dtype=jnp.bool_)

    # Step function for jax.lax.scan
    # jax.lax.scan is much more efficient than Python loops (ussually - depends on how you JIT)!
    # It compiles the loop body once and executes it repeatedly.
    def step_fn(carry, step_key):
        """Single rollout step with search-based action selection."""
        state, timestep, done = carry
        observation = timestep.observation

        # Early exit if all episodes are done (saves compute)
        all_done = jnp.all(done)

        def skip_step():
            """Return zeros when all episodes are done."""
            return (state, timestep, done), RolloutData(
                observation=observation,
                action_weights=jnp.zeros((batch_per_device, num_actions), dtype=jnp.float32),
                reward=jnp.zeros((batch_per_device,), dtype=jnp.float32),
                discount=jnp.zeros((batch_per_device,), dtype=jnp.float32),
                mask=jnp.zeros((batch_per_device,), dtype=jnp.bool_),
            )

        def do_step():
            """Run search and step the environment."""
            
            # eval network at current state
            (logits, value), _ = forward.apply(
                model_params, model_state, observation, is_eval=True
            )

            valid_mask = get_valid_action_mask(observation.action_mask)
            root_logits = apply_action_mask(logits, valid_mask)

            # Run Gumbel MuZero search
            # gumbel_muzero_policy: Uses Gumbel-Top-k sampling for exploration
            # and computes action_weights = softmax(prior + Q), a direct
            # policy improvement operator with finite-sample guarantees.
            # See: https://openreview.net/forum?id=bERaNdoegnO
            root = mctx.RootFnOutput(
                prior_logits=root_logits,
                value=value,
                embedding=state,  # State is the "embedding" for MCTS
            )

            policy_output = mctx.gumbel_muzero_policy(
                params=model,
                rng_key=step_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~valid_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )

            # Extract action and policy distribution
            action = policy_output.action
            action_weights = policy_output.action_weights.astype(jnp.float32)

            # Step env
            action_pair = unflatten_action(action)
            next_state, next_timestep = jax.vmap(env.step)(state, action_pair)

            # Only record data for active (non-terminated) episodes
            active = ~done
            reward = jnp.where(active, next_timestep.reward.astype(jnp.float32), 0.0)
            discount = jnp.where(
                active,
                next_timestep.discount.astype(jnp.float32) * config.discount,
                0.0,
            )
            action_weights = jnp.where(active[:, None], action_weights, 0.0)

            # Update done flags
            next_done = done | (next_timestep.discount == 0.0)

            return (next_state, next_timestep, next_done), RolloutData(
                observation=observation,
                action_weights=action_weights,
                reward=reward,
                discount=discount,
                mask=active,
            )

        return jax.lax.cond(all_done, skip_step, do_step)

    # episode loop
    step_keys = jax.random.split(scan_key, config.max_num_steps)
    (final_state, final_timestep, final_done), data = jax.lax.scan(
        step_fn, (state, timestep, done), step_keys
    )

    return data, final_timestep, final_done


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         Target Computation                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Compute Monte-Carlo returns as value targets.
# For BinPack with dense reward, the return is simply the sum of rewards
# (volume added at each step), which equals final utilization ∈ [0, 1].
#
# Note: The value head is trained from MC returns, not from search values.
# Only the policy is "search-improved"; the value learns from experience.

class TrainingSample(NamedTuple):
    """Processed sample ready for training."""
    observation: object          # Observation pytree
    policy_target: jnp.ndarray   # Search action distribution
    value_target: jnp.ndarray    # Monte-Carlo return
    mask: jnp.ndarray            # Valid sample indicator


@jax.pmap
def compute_value_targets(data: RolloutData) -> TrainingSample:
    """
    Compute Monte-Carlo value targets via reverse cumulative sum.

    For undiscounted episodic tasks (discount=1 until terminal):
        V(t) = r(t) + γ·r(t+1) + γ²·r(t+2) + ...

    We compute this efficiently using jax.lax.scan in reverse.

    Args:
        data: RolloutData with shape (T, B, ...)

    Returns:
        TrainingSample with value targets
    """
    # Reverse time dimension for cumulative sum
    rewards_reversed = data.reward[::-1]
    discounts_reversed = data.discount[::-1]

    def cumsum_fn(future_return, step_data):
        """Bellman backup: V(t) = r(t) + γ(t) * V(t+1)"""
        reward, discount = step_data
        current_return = reward + discount * future_return
        return current_return, current_return

    batch_size = data.reward.shape[1]
    _, returns_reversed = jax.lax.scan(
        cumsum_fn,
        jnp.zeros((batch_size,), dtype=jnp.float32),  # Terminal value = 0
        (rewards_reversed, discounts_reversed),
    )

    # Reverse back to forward time order
    value_targets = returns_reversed[::-1]

    return TrainingSample(
        observation=data.observation,
        policy_target=data.action_weights,
        value_target=value_targets,
        mask=data.mask,
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                            Train                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Loss function combines:
#   1. Policy loss: Cross-entropy between network output and search distribution
#   2. Value loss: L2 loss between predicted and Monte-Carlo return
#
# The policy loss is the key to search distillation: we train the network
# to imitate the search-improved policy via supervised learning.
# This is fundamentally different from policy gradients—reward doesn't
# appear in the policy loss at all!


def compute_loss(
    model_params,
    model_state,
    batch: TrainingSample
) -> Tuple[jnp.ndarray, Tuple[object, jnp.ndarray, jnp.ndarray]]:
    """
    Compute combined policy and value loss.

    Args:
        model_params: Network parameters
        model_state: Network state (e.g., BatchNorm stats)
        batch: TrainingSample with targets and mask

    Returns:
        total_loss: Scalar loss value
        aux: (new_model_state, policy_loss, value_loss)
    """
    (logits, value), model_state = forward.apply(
        model_params, model_state, batch.observation, is_eval=False
    )

    mask = batch.mask.astype(jnp.float32)
    num_valid = jnp.maximum(jnp.sum(mask), 1.0)  # Avoid division by zero

    # Policy Loss: Cross-entropy with search distribution
    # This is the distillation loss: teach the network to match search's policy
    # Note: This is imitation learning, not policy gradients!
    policy_loss_per_sample = optax.softmax_cross_entropy(logits, batch.policy_target)
    policy_loss = jnp.sum(policy_loss_per_sample * mask) / num_valid

    # Value Loss: L2 between prediction and Monte-Carlo return
    value_loss_per_sample = optax.l2_loss(value, batch.value_target)
    value_loss = jnp.sum(value_loss_per_sample * mask) / num_valid

    total_loss = policy_loss + config.value_loss_weight * value_loss

    return total_loss, (model_state, policy_loss, value_loss)


# Optimizer Setup
# We use Adam with gradient clipping for stable training.
# optax.chain composes multiple transforms.
# See: https://optax.readthedocs.io/en/latest/
# See my other optimizer for jax (not so subtle plug ik... - https://github.com/ComputationalRobotics/TRAC)

def create_optimizer():
    """Create optimizer with gradient clipping."""
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )


optimizer = create_optimizer()


@partial(jax.pmap, axis_name="devices")
def train_step(
    model,
    opt_state,
    batch: TrainingSample
) -> Tuple[object, object, jnp.ndarray, jnp.ndarray]:
    """
    Single training step with gradient synchronization across devices.

    The `axis_name="devices"` parameter enables jax.lax.pmean to average
    gradients across all devices, implementing data-parallel training.

    Args:
        model: (params, state) tuple, replicated across devices
        opt_state: Optimizer state, replicated
        batch: Training batch, sharded across devices

    Returns:
        new_model: Updated (params, state)
        new_opt_state: Updated optimizer state
        policy_loss: Averaged policy loss (for logging)
        value_loss: Averaged value loss (for logging)
    """
    model_params, model_state = model

    # Compute gradients
    grads, (new_state, policy_loss, value_loss) = jax.grad(
        compute_loss, has_aux=True
    )(model_params, model_state, batch)

    # Sync gradients across devices
    # jax.lax.pmean averages the gradients across all devices, implementing
    # synchronous data-parallel training.
    grads = jax.lax.pmean(grads, axis_name="devices")

    # Apply optimizer update
    updates, opt_state = optimizer.update(grads, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)

    # Average losses for logging
    policy_loss = jax.lax.pmean(policy_loss, axis_name="devices")
    value_loss = jax.lax.pmean(value_loss, axis_name="devices")

    return (model_params, new_state), opt_state, policy_loss, value_loss


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                            Eval                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Two evaluation modes:
#   1. Greedy: Take argmax of policy network (fast)
#   2. MCTS: Full search at each step (slower but stronger)
# During training, greedy evaluation is typically sufficient to track progress.


@jax.pmap
def evaluate_greedy(model, rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate policy using greedy action selection (no search).

    Args:
        model: (params, state) tuple
        rng_key: Random key for environment reset

    Returns:
        returns: Episode returns, shape (batch_per_device,)
        episode_lengths: Number of steps, shape (batch_per_device,)
    """
    model_params, model_state = model
    batch_per_device = config.eval_batch_size // num_devices

    # Reset environments
    reset_keys = jax.random.split(rng_key, batch_per_device)
    state, timestep = jax.vmap(env.reset)(reset_keys)

    # Initialize accumulators
    done = jnp.zeros((batch_per_device,), dtype=jnp.bool_)
    total_return = jnp.zeros((batch_per_device,), dtype=jnp.float32)
    episode_length = jnp.zeros((batch_per_device,), dtype=jnp.int32)

    def step_fn(carry, _):
        state, timestep, done, total_return, episode_length = carry
        observation = timestep.observation

        # Get policy logits and take greedy action
        (logits, _), _ = forward.apply(
            model_params, model_state, observation, is_eval=True
        )
        valid_mask = get_valid_action_mask(observation.action_mask)
        logits = apply_action_mask(logits, valid_mask)
        action = jnp.argmax(logits, axis=-1).astype(jnp.int32)

        # Step environment
        action_pair = unflatten_action(action)
        next_state, next_timestep = jax.vmap(env.step)(state, action_pair)

        # Accumulate returns and lengths for active episodes
        active = ~done
        reward = jnp.where(active, next_timestep.reward.astype(jnp.float32), 0.0)
        total_return = total_return + reward
        episode_length = episode_length + active.astype(jnp.int32)

        # Update done flags
        done = done | (next_timestep.discount == 0.0)

        return (next_state, next_timestep, done, total_return, episode_length), None

    (_, _, _, total_return, episode_length), _ = jax.lax.scan(
        step_fn,
        (state, timestep, done, total_return, episode_length),
        None,
        length=config.max_num_steps
    )

    return total_return, episode_length


@jax.pmap
def evaluate_mcts(model, rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate policy using MCTS at each decision point.

    This is slower but gives a better estimate of the policy's potential
    when combined with search.
    """
    model_params, model_state = model
    batch_per_device = config.eval_batch_size // num_devices

    rng_key, reset_key, scan_key = jax.random.split(rng_key, 3)
    reset_keys = jax.random.split(reset_key, batch_per_device)

    state, timestep = jax.vmap(env.reset)(reset_keys)
    done = jnp.zeros((batch_per_device,), dtype=jnp.bool_)
    total_return = jnp.zeros((batch_per_device,), dtype=jnp.float32)
    episode_length = jnp.zeros((batch_per_device,), dtype=jnp.int32)

    def step_fn(carry, step_key):
        state, timestep, done, total_return, episode_length = carry
        observation = timestep.observation

        # Network forward pass
        (logits, value), _ = forward.apply(
            model_params, model_state, observation, is_eval=True
        )
        valid_mask = get_valid_action_mask(observation.action_mask)
        root_logits = apply_action_mask(logits, valid_mask)

        # Run MCTS
        root = mctx.RootFnOutput(
            prior_logits=root_logits,
            value=value,
            embedding=state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=step_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.eval_num_simulations,
            invalid_actions=~valid_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )

        # Take action from MCTS
        action = policy_output.action.astype(jnp.int32)
        action_pair = unflatten_action(action)
        next_state, next_timestep = jax.vmap(env.step)(state, action_pair)

        # Accumulate
        active = ~done
        reward = jnp.where(active, next_timestep.reward.astype(jnp.float32), 0.0)
        total_return = total_return + reward
        episode_length = episode_length + active.astype(jnp.int32)
        done = done | (next_timestep.discount == 0.0)

        return (next_state, next_timestep, done, total_return, episode_length), None

    step_keys = jax.random.split(scan_key, config.max_num_steps)
    (_, _, _, total_return, episode_length), _ = jax.lax.scan(
        step_fn,
        (state, timestep, done, total_return, episode_length),
        step_keys
    )

    return total_return, episode_length


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                            Main  Loop                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# The Policy Iteration via Search Distillation algorithm:
#
#   for iteration in range(max_iterations):
#       1. COLLECT: Run episodes with search, record (obs, search_policy, reward)
#       2. COMPUTE: Calculate value targets (Monte-Carlo returns)
#       3. TRAIN: Update network to match search policy (cross-entropy)
#                 and value targets (MSE)
#       4. REPEAT: The improved network makes search even better next iteration
#
# This creates a virtuous cycle: better network → better search → better targets
# → even better network → ...
#
# Key insight: This is policy iteration where search is the improvement operator
# and supervised learning is how we internalize the improvement.


if __name__ == "__main__":
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.env_id}_nsim{config.num_simulations}_seed{config.seed}_{timestamp}"

    os.makedirs(config.save_dir, exist_ok=True)

    wandb.init(
        project=config.wandb_project,
        config=config.model_dump(),
        name=run_name,
        dir=config.save_dir,
    )

    # Master random key
    rng_key = jax.random.PRNGKey(config.seed)

    # Init Network
    # Haiku requires a sample input to trace the network and initialize params.
    rng_key, env_key, init_key = jax.random.split(rng_key, 3)

    # Create dummy observation for initialization
    dummy_state, dummy_timestep = env.reset(env_key)
    dummy_obs = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(x, axis=0),
        dummy_timestep.observation
    )

    params, net_state = forward.init(init_key, dummy_obs)
    model = (params, net_state)
    opt_state = optimizer.init(params)

    # Replicate Across Devices
    # jax.device_put_replicated copies data to all devices.
    # After this, model and opt_state have a leading device dimension.
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    # Ckpt dir structure
    ckpt_dir = os.path.join(
        config.save_dir,
        config.env_id,
        f"nsim_{config.num_simulations}",
        f"seed_{config.seed}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training stats
    iteration = 0
    total_frames = 0.0
    total_hours = 0.0

    print("=" * 60)
    print("Starting Policy Iteration via Search Distillation")
    print("=" * 60)

    # Main Loop
    while True:
        # Eval
        if iteration % config.eval_interval == 0:
            rng_key, eval_key = jax.random.split(rng_key)
            eval_keys = jax.random.split(eval_key, num_devices)

            if config.eval_use_mcts:
                returns, lengths = evaluate_mcts(model, eval_keys)
                eval_tag = "eval/mcts"
            else:
                returns, lengths = evaluate_greedy(model, eval_keys)
                eval_tag = "eval/greedy"

            avg_return = float(returns.mean())
            avg_length = float(lengths.mean())

            wandb.log({
                "iteration": iteration,
                f"{eval_tag}/avg_return": avg_return,
                f"{eval_tag}/avg_steps": avg_length,
                "hours": total_hours,
                "frames": total_frames,
            })
            print(f"[Eval] iter={iteration:4d} | {eval_tag} | "
                  f"return={avg_return:.4f} | steps={avg_length:.1f}")

        if iteration % config.save_interval == 0:
            # Extract single-device copy for saving
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
                    "jumanji_version": getattr(jumanji, "__version__", "unknown"),
                }, f)
            print(f"[Checkpoint] Saved: {ckpt_path}")

        # Check termination
        if iteration >= config.max_num_iters:
            print("=" * 60)
            print("Training Complete!")
            print("=" * 60)
            break

        # STEP 1: Collect exp
        # Run search-guided rollouts across all devices in parallel.
        # Each device handles (rollout_batch_size / num_devices) episodes.
        iter_start = time.time()

        rng_key, collect_key = jax.random.split(rng_key)
        collect_keys = jax.random.split(collect_key, num_devices)

        rollout_data, _, _ = collect_experience(model, collect_keys)

        # STEP 2: Compute Value targets
        training_samples = compute_value_targets(rollout_data)

        rollout_data_host = jax.device_get(rollout_data)
        episode_returns = rollout_data_host.reward.sum(axis=1)  # Sum over time
        episode_lengths = rollout_data_host.mask.sum(axis=1)
        avg_rollout_return = float(episode_returns.mean())
        avg_rollout_length = float(episode_lengths.mean())

        total_frames += float(rollout_data_host.mask.sum())

        # STEP 3: Train
        # Shuffle samples and train in minibatches.
        samples_host = jax.device_get(training_samples)

        # Flatten: (num_devices, time, batch_per_device, ...) → (N, ...)
        def flatten_samples(x):
            return x.reshape((-1,) + x.shape[3:])

        flat_samples = jax.tree_util.tree_map(flatten_samples, samples_host)

        total_samples = flat_samples.mask.shape[0]
        num_updates = total_samples // config.training_batch_size
        samples_used = num_updates * config.training_batch_size

        if num_updates == 0:
            print(f"[Warning] Not enough samples for update: "
                  f"{total_samples} < {config.training_batch_size}")
            iteration += 1
            continue

        # Shuffle samples
        rng_key, shuffle_key = jax.random.split(rng_key)
        permutation = jax.device_get(
            jax.random.permutation(shuffle_key, jnp.arange(samples_used))
        )

        flat_samples = jax.tree_util.tree_map(
            lambda x: x[:samples_used][permutation],
            flat_samples
        )

        # Reshape into minibatches: (num_updates, num_devices, batch_per_device, ...)
        batch_per_device = config.training_batch_size // num_devices

        def to_minibatches(x):
            return x.reshape((num_updates, num_devices, batch_per_device) + x.shape[1:])

        minibatches = jax.tree_util.tree_map(to_minibatches, flat_samples)

        # Training loop over minibatches
        policy_losses = []
        value_losses = []

        for update_idx in range(num_updates):
            batch = jax.tree_util.tree_map(lambda x: x[update_idx], minibatches)
            model, opt_state, policy_loss, value_loss = train_step(model, opt_state, batch)

            # Extract from first device for logging
            policy_losses.append(float(jax.device_get(policy_loss[0])))
            value_losses.append(float(jax.device_get(value_loss[0])))

        avg_policy_loss = sum(policy_losses) / len(policy_losses)
        avg_value_loss = sum(value_losses) / len(value_losses)
        
        iter_time = time.time() - iter_start
        total_hours += iter_time / 3600.0

        metrics = {
            "iteration": iteration,
            "hours": total_hours,
            "frames": total_frames,
            "rollout/avg_return": avg_rollout_return,
            "rollout/avg_steps": avg_rollout_length,
            "train/policy_loss": avg_policy_loss,
            "train/value_loss": avg_value_loss,
            "search/num_simulations": config.num_simulations,
        }
        wandb.log(metrics)

        print(f"[Train] iter={iteration:4d} | "
              f"return={avg_rollout_return:.4f} | "
              f"policy_loss={avg_policy_loss:.4f} | "
              f"value_loss={avg_value_loss:.4f} | "
              f"time={iter_time:.1f}s")

        iteration += 1

    wandb.finish()
