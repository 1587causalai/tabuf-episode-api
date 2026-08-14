"""Observational v0 TabUF episode generator (DiscoSCM-aligned factor law).

This samples Unit×Feature grids for TabUF training. It is observational v0
only: no intervention, no counterfactual noise redraw. It is not a DiscoSCM
paper reimplementation of Layer 3.

Generative law
--------------
Population: n realized units with attributes U of shape (n, k), U ~ N(0, I).
Shared feature directions W of shape (d, k), W ~ N(0, I), then optional
column-normalize so each of the k latent axes has unit L2 length in R^d.
Observational noise E ~ N(0, sigma^2).

    Y[i, j] = <U[i], W[j]> + E[i, j]

Row i is an observation attributed to realized unit u_i. Do not read
U[i] ~ N(0, I) as "each row is a new population": the population is the
set of n realized units {u_0, ..., u_{n-1}} that share feature directions W.

No natural missingness. Query set Q is a random subset of cells (~query_frac);
context C is the complement. C ∩ Q = empty, C ∪ Q = all cells.

The model must not receive U or W unless debug=True.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_UNIT_DIM = 4
DEFAULT_SIGMA = 0.3
DEFAULT_QUERY_FRAC = 0.15


def _json_float_grid(arr: np.ndarray) -> list[list[float | None]]:
    """Nested lists; NaN becomes JSON null."""
    out: list[list[float | None]] = []
    for row in arr:
        out.append([None if not np.isfinite(v) else float(v) for v in row])
    return out


def _json_bool_grid(arr: np.ndarray) -> list[list[bool]]:
    return arr.astype(bool).tolist()


def sample_episode(
    rng: np.random.Generator,
    *,
    n_units: int = 64,
    n_features: int = 8,
    unit_dim: int = DEFAULT_UNIT_DIM,
    query_frac: float = DEFAULT_QUERY_FRAC,
    sigma: float = DEFAULT_SIGMA,
    column_normalize: bool = True,
    debug: bool = False,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Draw one observational Unit×Feature episode.

    Parameters
    ----------
    n_units, n_features, unit_dim
        Grid size n×d and latent unit dim k.
    query_frac
        Fraction of all cells assigned to Q (the rest are C).
    sigma
        Observational noise std.
    column_normalize
        If True, L2-normalize columns of W.
    debug
        If True, attach latent U, W and full Y. Default payloads omit them.
    seed
        Episode seed recorded in the payload (sampling uses `rng`).
    """
    n = int(n_units)
    d = int(n_features)
    k = int(unit_dim)

    # Realized units of one population. Row i ↔ unit u_i, not a new population.
    U = rng.standard_normal((n, k))
    W = rng.standard_normal((d, k))
    if column_normalize:
        col_norm = np.linalg.norm(W, axis=0, keepdims=True)
        W = W / np.maximum(col_norm, 1e-8)

    E = rng.normal(0.0, float(sigma), size=(n, d))
    # Y[i, j] = <U[i], W[j]> ; W[j] is row j of W.
    Y = U @ W.T + E

    n_cells = n * d
    n_query = int(round(float(query_frac) * n_cells))
    n_query = max(1, min(n_query, n_cells - 1))
    pick = rng.choice(n_cells, size=n_query, replace=False)
    query_mask = np.zeros(n_cells, dtype=bool)
    query_mask[pick] = True
    query_mask = query_mask.reshape(n, d)
    context_mask = ~query_mask

    y_input = Y.copy()
    y_input[query_mask] = np.nan
    # Flattened in C-order: y_query[t] is the t-th True of query_mask.ravel().
    y_query = Y[query_mask].astype(np.float64)

    n_q = int(query_mask.sum())
    n_c = int(context_mask.sum())
    payload: dict[str, Any] = {
        "context_mask": _json_bool_grid(context_mask),
        "query_mask": _json_bool_grid(query_mask),
        "y_input": _json_float_grid(y_input),
        "y_query": [float(v) for v in y_query],
        "shapes": {
            "n_units": n,
            "n_features": d,
            "unit_dim": k,
            "n_query": n_q,
            "n_context": n_c,
            "y_input": [n, d],
            "context_mask": [n, d],
            "query_mask": [n, d],
            "y_query": [n_q],
        },
        "seed": seed,
    }
    if debug:
        payload["U"] = _json_float_grid(U)
        payload["W"] = _json_float_grid(W)
        payload["Y_full"] = _json_float_grid(Y)
    return payload


def sample_episodes(
    *,
    n_units: int = 64,
    n_features: int = 8,
    unit_dim: int = DEFAULT_UNIT_DIM,
    query_frac: float = DEFAULT_QUERY_FRAC,
    sigma: float = DEFAULT_SIGMA,
    seed: int | None = 0,
    n_episodes: int = 1,
    debug: bool = False,
    column_normalize: bool = True,
) -> list[dict[str, Any]]:
    """Draw n_episodes independent observational grids.

    If `seed` is an int, episode e uses seed+e. If `seed` is None, a random
    base seed is drawn and still recorded per episode so the draw is named.
    """
    n_episodes = max(1, min(int(n_episodes), 32))
    if seed is None:
        base = int(np.random.default_rng().integers(0, 2**31 - 1))
    else:
        base = int(seed)

    episodes: list[dict[str, Any]] = []
    for e in range(n_episodes):
        ep_seed = base + e
        rng = np.random.default_rng(ep_seed)
        episodes.append(
            sample_episode(
                rng,
                n_units=n_units,
                n_features=n_features,
                unit_dim=unit_dim,
                query_frac=query_frac,
                sigma=sigma,
                column_normalize=column_normalize,
                debug=debug,
                seed=ep_seed,
            )
        )
    return episodes
