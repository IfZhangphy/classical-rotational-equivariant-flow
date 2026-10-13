"""Resource-aware rotationally equivariant normalizing flow.

The model targets a two-dimensional classical Coulomb gas in a harmonic
potential.  It uses alternating affine coupling layers.  The conditioner
only builds scalar features from squared distances and constructs vector
shifts as weighted sums of condition particle positions, so a global planar
rotation rotates the output in exactly the same way.

The defaults are sized for an 8 GB GPU.  Run a short validation pass with::

    python rotational_equivariant_flow.py --smoke-test

Then start a small training run with::

    python rotational_equivariant_flow.py --epochs 2000 --batch-size 256
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Avoid reserving all VRAM before JAX knows the actual working set.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")

import haiku as hk
import jax

# Disable TF32-style reduced precision on GPU matrix multiplications.  The
# equivariance check compares two mathematically identical paths, so using a
# lower-precision GEMM can make a correct model appear non-equivariant.
jax.config.update("jax_default_matmul_precision", "float32")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax


LOG_2PI = np.log(2.0 * np.pi)


@dataclass(frozen=True)
class Config:
    n_particles: int = 20
    dim: int = 2
    beta: float = 10.0
    hidden_dim: int = 48
    message_steps: int = 2
    num_layers: int = 4
    batch_size: int = 256
    epochs: int = 2000
    learning_rate: float = 1e-3
    scale_limit: float = 0.50
    shift_limit: float = 1.00
    # A small soft core prevents rare near-collisions from dominating the
    # first optimization steps.  The physical 1/r behavior is unchanged at
    # distances relevant to the learned shell; lower it only for final audits.
    distance_floor: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42
    x64: bool = False
    plot_every: int = 100
    output_dir: str = "rotational_equivariant_results"


def _layer_indices(n_particles: int, layer: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return condition and target particle indices for one coupling layer."""
    even = tuple(range(0, n_particles, 2))
    odd = tuple(range(1, n_particles, 2))
    if layer % 2 == 0:
        return even, odd
    return odd, even


class EquivariantConditioner(hk.Module):
    """Generate invariant scales and equivariant vector shifts.

    The conditioner sees only the condition particles.  Its scalar pathway
    uses squared distances and squared radii.  Its vector pathway is a
    weighted sum of condition coordinates, which is covariant under SO(2).
    Consequently the resulting shift is independent of the target
    coordinates and the affine coupling remains exactly invertible.
    """

    def __init__(
        self,
        cond_indices: tuple[int, ...],
        target_indices: tuple[int, ...],
        hidden_dim: int,
        message_steps: int,
        scale_limit: float,
        shift_limit: float,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.cond_indices = cond_indices
        self.target_indices = target_indices
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        self.scale_limit = scale_limit
        self.shift_limit = shift_limit
        self.pair_mlp = hk.nets.MLP([hidden_dim, hidden_dim], activation=jax.nn.silu)
        self.node_mlp = hk.nets.MLP([hidden_dim, hidden_dim], activation=jax.nn.silu)
        self.node_update_mlp = hk.nets.MLP([hidden_dim, hidden_dim], activation=jax.nn.silu)
        self.global_mlp = hk.nets.MLP([hidden_dim, hidden_dim], activation=jax.nn.silu)
        self.output_mlp = hk.nets.MLP([hidden_dim, hidden_dim, 3], activation=jax.nn.silu)
        self.edge_mlp = hk.nets.MLP([hidden_dim, hidden_dim, 1], activation=jax.nn.silu)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        cond = x[jnp.asarray(self.cond_indices, dtype=jnp.int32)]
        target_count = len(self.target_indices)
        cond_count = len(self.cond_indices)

        # Every input to the scalar pathway is invariant under a global
        # rotation: distances and radii are unchanged.
        radius2 = jnp.sum(cond * cond, axis=-1, keepdims=True)
        delta = cond[:, None, :] - cond[None, :, :]
        distance2 = jnp.sum(delta * delta, axis=-1, keepdims=True)
        pair_features = jnp.concatenate(
            [
                distance2,
                jnp.broadcast_to(radius2[:, None, :], (cond_count, cond_count, 1)),
                jnp.broadcast_to(radius2[None, :, :], (cond_count, cond_count, 1)),
            ],
            axis=-1,
        )
        pair_hidden = self.pair_mlp(pair_features)
        node_hidden = jnp.mean(pair_hidden, axis=1)
        node_input = jnp.concatenate([node_hidden, radius2], axis=-1)
        node_hidden = self.node_mlp(node_input)

        for _ in range(self.message_steps - 1):
            node_hidden = node_hidden + self.node_update_mlp(node_hidden)

        global_input = jnp.concatenate(
            [jnp.mean(node_hidden, axis=0), jnp.mean(radius2, axis=0)], axis=-1
        )
        global_hidden = self.global_mlp(global_input)

        target_embedding = hk.get_parameter(
            "target_embedding",
            shape=(target_count, self.hidden_dim),
            init=hk.initializers.TruncatedNormal(stddev=0.02),
        )
        target_context = jnp.concatenate(
            [jnp.broadcast_to(global_hidden, (target_count, self.hidden_dim)), target_embedding],
            axis=-1,
        )
        outputs = self.output_mlp(target_context)
        scale = self.scale_limit * jnp.tanh(outputs[:, 0])

        # Each target gets a separate invariant weight for every condition
        # particle.  The weighted coordinate sum is therefore expressive and
        # remains exactly SO(2)-covariant.
        edge_features = jnp.concatenate(
            [
                jnp.broadcast_to(node_hidden[None, :, :], (target_count, cond_count, self.hidden_dim)),
                jnp.broadcast_to(target_embedding[:, None, :], (target_count, cond_count, self.hidden_dim)),
            ],
            axis=-1,
        )
        edge_weights = self.edge_mlp(edge_features)[..., 0]
        edge_weights = self.shift_limit * jnp.tanh(edge_weights)
        shift = jnp.einsum("tc,cd->td", edge_weights, cond)
        return scale, shift


def make_flow(config: Config) -> tuple[Any, Any]:
    """Create forward and inverse Haiku transforms sharing parameter names."""

    def flow_apply(z: jax.Array, inverse: bool) -> tuple[jax.Array, jax.Array]:
        x = jnp.reshape(z, (config.n_particles, config.dim))
        log_det = jnp.asarray(0.0, dtype=z.dtype)
        layer_order = range(config.num_layers - 1, -1, -1) if inverse else range(config.num_layers)

        for layer in layer_order:
            cond_indices, target_indices = _layer_indices(config.n_particles, layer)
            conditioner = EquivariantConditioner(
                cond_indices=cond_indices,
                target_indices=target_indices,
                hidden_dim=config.hidden_dim,
                message_steps=config.message_steps,
                scale_limit=config.scale_limit,
                shift_limit=config.shift_limit,
                name=f"conditioner_{layer}",
            )
            scale, shift = conditioner(x)
            target = x[jnp.asarray(target_indices, dtype=jnp.int32)]
            if inverse:
                target = (target - shift) * jnp.exp(-scale[:, None])
                log_det = log_det - jnp.sum(scale)
            else:
                target = target * jnp.exp(scale[:, None]) + shift
                log_det = log_det + jnp.sum(scale)
            x = x.at[jnp.asarray(target_indices, dtype=jnp.int32)].set(target)

        return jnp.reshape(x, (-1,)), log_det

    @hk.without_apply_rng
    @hk.transform
    def forward(z: jax.Array) -> tuple[jax.Array, jax.Array]:
        return flow_apply(z, inverse=False)

    @hk.without_apply_rng
    @hk.transform
    def inverse(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return flow_apply(x, inverse=True)

    return forward, inverse


def energy_fn(
    x: jax.Array, n_particles: int, dim: int, distance_floor: float = 1e-3
) -> jax.Array:
    """Harmonic confinement plus pairwise Coulomb repulsion."""
    positions = jnp.reshape(x, (n_particles, dim))
    i, j = np.triu_indices(n_particles, k=1)
    differences = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(differences * differences, axis=-1) + distance_floor**2)
    return jnp.sum(positions * positions) + jnp.sum(1.0 / distances)


def _make_train_step(flow: Any, config: Config) -> Any:
    batch_forward = jax.vmap(flow.apply, in_axes=(None, 0), out_axes=(0, 0))
    batch_energy = jax.vmap(
        lambda x: energy_fn(x, config.n_particles, config.dim, config.distance_floor), in_axes=0
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )

    def loss_fn(params: hk.Params, z_batch: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        x_batch, log_det = batch_forward(params, z_batch)
        log_p_base = -0.5 * jnp.sum(z_batch * z_batch + LOG_2PI, axis=-1)
        log_p_x = log_p_base - log_det
        energies = batch_energy(x_batch)
        free_energy = log_p_x / config.beta + energies
        return jnp.mean(free_energy), (
            jnp.std(free_energy),
            jnp.mean(energies),
            jnp.mean(log_det),
        )

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def step(
        params: hk.Params, opt_state: optax.OptState, z_batch: jax.Array
    ) -> tuple[hk.Params, optax.OptState, jax.Array, tuple[jax.Array, ...]]:
        (loss, aux), grads = value_and_grad(params, z_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    return step, optimizer


def equivariance_error(
    flow: Any, params: hk.Params, config: Config, key: jax.Array, count: int = 16
) -> float:
    """Measure max absolute error of f(Rz) = Rf(z)."""
    z = jax.random.normal(key, (count, config.n_particles * config.dim))
    theta = jnp.asarray(0.731, dtype=z.dtype)
    rotation = jnp.asarray(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )
    z_rotated = jnp.reshape(z, (count, config.n_particles, config.dim)) @ rotation.T
    z_rotated = jnp.reshape(z_rotated, (count, -1))
    x = jax.vmap(flow.apply, in_axes=(None, 0), out_axes=(0, 0))(params, z)[0]
    x_rotated = jax.vmap(flow.apply, in_axes=(None, 0), out_axes=(0, 0))(params, z_rotated)[0]
    expected = jnp.reshape(x, (count, config.n_particles, config.dim)) @ rotation.T
    return float(jnp.max(jnp.abs(x_rotated - jnp.reshape(expected, (count, -1)))))


def inverse_error(
    forward: Any, inverse: Any, params: hk.Params, config: Config, key: jax.Array, count: int = 16
) -> float:
    z = jax.random.normal(key, (count, config.n_particles * config.dim))
    x = jax.vmap(forward.apply, in_axes=(None, 0), out_axes=(0, 0))(params, z)[0]
    z_reconstructed = jax.vmap(inverse.apply, in_axes=(None, 0), out_axes=(0, 0))(params, x)[0]
    return float(jnp.max(jnp.abs(z - z_reconstructed)))


def train(config: Config) -> tuple[Any, Any, hk.Params, np.ndarray]:
    if config.x64:
        jax.config.update("jax_enable_x64", True)
    if config.dim != 2:
        raise ValueError("当前等变向量构造针对二维旋转群，dim 必须为 2。")
    if config.n_particles < 4 or config.n_particles % 2:
        raise ValueError("n_particles 必须是大于等于 4 的偶数。")
    if config.num_layers < 2:
        raise ValueError("num_layers 至少需要 2 层。")
    if config.message_steps < 1:
        raise ValueError("message_steps 至少需要 1 次。")
    if config.batch_size < 1:
        raise ValueError("batch_size 必须为正数。")

    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    forward, inverse = make_flow(config)
    dummy = jnp.zeros((config.n_particles * config.dim,), dtype=jnp.float64 if config.x64 else jnp.float32)
    params = forward.init(init_key, dummy)

    step, optimizer = _make_train_step(forward, config)
    opt_state = optimizer.init(params)
    history: list[list[float]] = []
    start = time.perf_counter()

    for epoch in range(config.epochs):
        key, batch_key = jax.random.split(key)
        z_batch = jax.random.normal(
            batch_key,
            (config.batch_size, config.n_particles * config.dim),
            dtype=dummy.dtype,
        )
        params, opt_state, loss, aux = step(params, opt_state, z_batch)
        loss_value = float(loss)
        std_value = float(aux[0])
        history.append([loss_value, std_value / np.sqrt(config.batch_size)])

        if config.plot_every and (epoch == 0 or (epoch + 1) % config.plot_every == 0):
            elapsed = time.perf_counter() - start
            print(
                f"epoch={epoch + 1:5d} loss={loss_value: .6f} "
                f"energy={float(aux[1]): .6f} logdet={float(aux[2]): .6f} "
                f"time={elapsed: .1f}s"
            )

    return forward, inverse, params, np.asarray(history, dtype=np.float64)


def save_checkpoint(path: Path, config: Config, params: hk.Params, history: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {"config": asdict(config), "params": jax.device_get(params), "history": history},
            handle,
        )


def save_plot(path: Path, history: np.ndarray, samples: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = samples.reshape(-1, 2)
    radii = np.sqrt(np.sum(points * points, axis=-1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist2d(points[:, 0], points[:, 1], bins=80, range=[[-4, 4], [-4, 4]], cmap="magma")
    axes[0].set_title("Generated particle density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].hist(radii, bins=80, range=(0, 4), density=True, color="tab:orange")
    axes[1].set_title("Radial density P(r)")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.25)
    axes[2].plot(history[:, 0], color="tab:blue")
    axes[2].set_title("Variational free energy")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--message-steps", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--scale-limit", type=float, default=None)
    parser.add_argument("--shift-limit", type=float, default=None)
    parser.add_argument("--distance-floor", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot-every", type=int, default=None)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="运行 2 个小批次并检查等变性与可逆性")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        epochs=2 if args.smoke_test else (args.epochs or Config.epochs),
        batch_size=16 if args.smoke_test else (args.batch_size or Config.batch_size),
        hidden_dim=16 if args.smoke_test else (args.hidden_dim or Config.hidden_dim),
        message_steps=1 if args.smoke_test else (args.message_steps or Config.message_steps),
        num_layers=2 if args.smoke_test else (args.layers or Config.num_layers),
        learning_rate=args.learning_rate or Config.learning_rate,
        scale_limit=args.scale_limit or Config.scale_limit,
        shift_limit=args.shift_limit or Config.shift_limit,
        distance_floor=args.distance_floor or Config.distance_floor,
        grad_clip=args.grad_clip or Config.grad_clip,
        seed=args.seed or Config.seed,
        x64=args.x64,
        plot_every=0 if args.smoke_test else (args.plot_every or Config.plot_every),
        output_dir=args.output_dir or Config.output_dir,
    )
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(f"config={asdict(config)}")
    forward, inverse, params, history = train(config)
    key = jax.random.PRNGKey(config.seed + 1)
    inv_error = inverse_error(forward, inverse, params, config, key)
    eq_error = equivariance_error(forward, params, config, key)
    print(f"inverse_max_abs_error={inv_error:.3e}")
    print(f"equivariance_max_abs_error={eq_error:.3e}")
    if args.smoke_test:
        if inv_error > 1e-5 or eq_error > 1e-5:
            raise RuntimeError("等变性或可逆性检查未通过。")
        print("smoke test passed")
        return

    output_dir = Path(config.output_dir)
    save_checkpoint(output_dir / "checkpoint.pkl", config, params, history)
    samples = jax.vmap(forward.apply, in_axes=(None, 0), out_axes=(0, 0))(
        params, jax.random.normal(key, (1024, config.n_particles * config.dim))
    )[0]
    save_plot(output_dir / "training.png", np.asarray(history), np.asarray(samples))
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
