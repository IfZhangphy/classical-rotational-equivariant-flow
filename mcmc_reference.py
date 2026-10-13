"""Local MCMC reference sampler for the beta=10 Coulomb gas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_energy(n_particles: int, dim: int, distance_floor: float):
    i, j = np.triu_indices(n_particles, k=1)
    i = jnp.asarray(i, dtype=jnp.int32)
    j = jnp.asarray(j, dtype=jnp.int32)

    @jax.jit
    def energy_batch(x: jax.Array) -> jax.Array:
        positions = jnp.reshape(x, (-1, n_particles, dim))
        differences = positions[:, i, :] - positions[:, j, :]
        distances = jnp.sqrt(
            jnp.sum(differences * differences, axis=-1)
            + distance_floor**2
        )
        return jnp.sum(positions * positions, axis=(1, 2)) + jnp.sum(
            1.0 / distances, axis=-1
        )

    return energy_batch


def run_mcmc(
    *,
    chains: int,
    n_particles: int,
    dim: int,
    beta: float,
    steps: int,
    proposal_width: float,
    seed: int,
    distance_floor: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    energy_batch = make_energy(n_particles, dim, distance_floor)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    x0 = jax.random.normal(
        init_key,
        (chains, n_particles, dim),
        dtype=jnp.float64,
    )

    @jax.jit
    def sample(x_init: jax.Array, initial_key: jax.Array):
        logp_init = -beta * energy_batch(x_init)

        def body(_, state):
            x, logp, key, accepted = state
            key, proposal_key, accept_key = jax.random.split(key, 3)
            proposal = x + proposal_width * jax.random.normal(
                proposal_key, x.shape, dtype=x.dtype
            )
            proposal_logp = -beta * energy_batch(proposal)
            log_alpha = jnp.minimum(0.0, proposal_logp - logp)
            log_u = jnp.log(
                jax.random.uniform(
                    accept_key,
                    shape=(x.shape[0],),
                    minval=1e-12,
                    dtype=x.dtype,
                )
            )
            take = log_u < log_alpha
            x = jnp.where(take[:, None, None], proposal, x)
            logp = jnp.where(take, proposal_logp, logp)
            return x, logp, key, accepted + jnp.sum(take)

        x, _, _, accepted = jax.lax.fori_loop(
            0,
            steps,
            body,
            (x_init, logp_init, initial_key, jnp.asarray(0.0)),
        )
        return x, accepted / (steps * x.shape[0])

    samples, acceptance = sample(x0, key)
    samples = np.asarray(jax.device_get(samples))
    acceptance = float(acceptance)
    soft_energy = np.asarray(jax.device_get(energy_batch(samples)))
    exact_energy = np.asarray(
        jax.device_get(make_energy(n_particles, dim, 0.0)(samples))
    )
    return samples, acceptance, soft_energy, exact_energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--proposal-width", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--distance-floor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("mcmc_beta10_reference"))
    args = parser.parse_args()

    samples, acceptance, soft_energy, exact_energy = run_mcmc(
        chains=args.chains,
        n_particles=20,
        dim=2,
        beta=args.beta,
        steps=args.steps,
        proposal_width=args.proposal_width,
        seed=args.seed,
        distance_floor=args.distance_floor,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "samples.npz",
        samples=samples,
        soft_energy=soft_energy,
        exact_energy=exact_energy,
    )

    points = samples.reshape(-1, 2)
    radii = np.linalg.norm(points, axis=-1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist2d(
        points[:, 0], points[:, 1], bins=80, range=[[-4, 4], [-4, 4]], cmap="magma"
    )
    axes[0].set_title("MCMC particle density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].hist(radii, bins=80, range=(0, 4), density=True, color="tab:green")
    axes[1].set_title("MCMC radial density P(r)")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.25)
    axes[2].hist(soft_energy, bins=60, alpha=0.75, label="soft floor")
    axes[2].hist(exact_energy, bins=60, alpha=0.55, label="exact 1/r")
    axes[2].set_title("MCMC energy")
    axes[2].set_xlabel("energy")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "mcmc_reference.png", dpi=140)
    plt.close(fig)

    summary = {
        "beta": args.beta,
        "temperature": 1.0 / args.beta,
        "distance_floor": args.distance_floor,
        "chains": args.chains,
        "steps": args.steps,
        "proposal_width": args.proposal_width,
        "acceptance": acceptance,
        "soft_energy_mean": float(soft_energy.mean()),
        "soft_energy_sem": float(soft_energy.std(ddof=1) / np.sqrt(len(soft_energy))),
        "exact_energy_mean": float(exact_energy.mean()),
        "exact_energy_sem": float(exact_energy.std(ddof=1) / np.sqrt(len(exact_energy))),
        "radius_mean": float(radii.mean()),
        "radius_std": float(radii.std(ddof=1)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"saved={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
