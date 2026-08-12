"""Synthetic graph generators with planted label mechanisms.

First generator (synthetic experiment 1): `rgg_homophily_graph` plants the
homophily-reach mechanism behind the Amazon Ratings regime. A random
geometric graph on the unit torus (polynomial neighborhood growth: bags stay
far from graph coverage at every tested radius, so no saturation confound),
with labels drawn from a smooth random field whose wavelength sets the
homophily correlation length, and an optional faint coupling that lets the
field leave a weak trace in node connectivity.

Dials:
- `wavelength`: spatial correlation length of the label field. Longer
  wavelength -> label homophily reaches further, in hops.
- `coupling` (beta): per-node connection radius scaled by (1 + beta * field).
  beta = 0 is the negative control: labels leave NO trace in structure, so
  every structural method must sit at the permutation floor.

Output matches the catalog CSV shape (edges_df: src_node_id/dest_node_id;
nodes_df: node_id + label column) so the standard session pipeline consumes
generated graphs unchanged.
"""

import math

import numpy as np
import pandas as pd

LABEL_COLUMN = "field_class"


def _torus_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Coordinate-wise minimum-image differences on the unit torus."""
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


def _smooth_field(points: np.ndarray, wavelength: float, rng: np.random.RandomState, n_components: int = 6) -> np.ndarray:
    """Smooth random field: sum of random-direction Fourier components.

    Frequencies are rounded to integers per axis so the field is continuous
    across the torus seams; magnitudes target the requested wavelength.
    """
    freq = max(1, round(1.0 / wavelength))
    field = np.zeros(len(points))
    for _ in range(n_components):
        theta = rng.uniform(0, 2 * np.pi)
        kx = round(freq * math.cos(theta))
        ky = round(freq * math.sin(theta))
        if kx == 0 and ky == 0:
            kx = freq
        phase = rng.uniform(0, 2 * np.pi)
        field += rng.normal(0, 1) * np.sin(2 * np.pi * (kx * points[:, 0] + ky * points[:, 1]) + phase)
    field = (field - field.mean()) / field.std()
    return field


def rgg_homophily_graph(
    n: int = 6000,
    mean_degree: float = 8.0,
    wavelength: float = 0.15,
    coupling: float = 0.0,
    n_classes: int = 5,
    label_noise: float = 0.1,
    seed: int = 13,
    chunk: int = 512,
):
    """Random geometric graph on the unit torus with a planted smooth label field.

    Args:
        n: number of nodes.
        mean_degree: target mean degree (sets the base connection radius
            r0 = sqrt(mean_degree / (pi * n))).
        wavelength: spatial wavelength of the label field (homophily dial).
        coupling: faint feature-label coupling beta; per-node radius is
            r0 * (1 + beta * field), clipped at +-0.5 modulation. 0 = none.
        n_classes: number of label classes (field quantiles).
        label_noise: probability a node's label is replaced by a uniform
            random class (keeps the ceiling honest).
        seed: RNG seed; fully deterministic per seed.
        chunk: row-block size for the pairwise distance sweep.

    Returns:
        (edges_df, nodes_df): catalog-shaped frames; nodes_df carries
        LABEL_COLUMN with integer classes 0..n_classes-1.
    """
    if not 0 <= coupling <= 0.5:
        raise ValueError(f"coupling must be in [0, 0.5], got {coupling}")
    rng = np.random.RandomState(seed)
    points = rng.uniform(0, 1, size=(n, 2))
    field = _smooth_field(points, wavelength, rng)

    r0 = math.sqrt(mean_degree / (math.pi * n))
    radii = r0 * np.clip(1.0 + coupling * field, 0.5, 1.5)

    # Edge if torus distance < average of the two nodes' radii.
    src, dst = [], []
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        dx = _torus_delta(points[i0:i1, None, 0], points[None, :, 0])
        dy = _torus_delta(points[i0:i1, None, 1], points[None, :, 1])
        dist = np.sqrt(dx * dx + dy * dy)
        thresh = (radii[i0:i1, None] + radii[None, :]) / 2.0
        rows, cols = np.nonzero(dist < thresh)
        rows = rows + i0
        keep = rows < cols  # upper triangle only, no self-loops
        src.append(rows[keep])
        dst.append(cols[keep])
    edges_df = pd.DataFrame({"src_node_id": np.concatenate(src), "dest_node_id": np.concatenate(dst)})

    # Labels: field quantiles + flip noise.
    quantiles = np.quantile(field, np.linspace(0, 1, n_classes + 1)[1:-1])
    labels = np.digitize(field, quantiles)
    flip = rng.uniform(size=n) < label_noise
    labels[flip] = rng.randint(0, n_classes, size=flip.sum())

    nodes_df = pd.DataFrame({"node_id": np.arange(n), LABEL_COLUMN: labels})
    return edges_df, nodes_df
