"""Observational v0 TabUF episode generator (DiscoSCM-aligned).

Return unit is an episode: one Unit x Feature table with three disjoint masks
(missing / context / query). Behind each table sits a population of units and a
unit-specific response law. Those are omitted unless return_mechanism=True.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_UNIT_DIM = 4
DEFAULT_SIGMA = 0.3
DEFAULT_QUERY_FRAC = 0.15
DEFAULT_MISSING_FRAC = 0.0
MECHANISM_LAW = "Y[i,j] = <U[i], W[j]> + E[i,j]"


def _json_float_grid(arr: np.ndarray) -> list[list[float | None]]:
    out: list[list[float | None]] = []
    for row in arr:
        out.append([None if not np.isfinite(v) else float(v) for v in row])
    return out


def _json_bool_grid(arr: np.ndarray) -> list[list[bool]]:
    return arr.astype(bool).tolist()


def _pack_population(U: np.ndarray, *, seed: int | None) -> dict[str, Any]:
    n, k = int(U.shape[0]), int(U.shape[1])
    return {
        "id": f"pop-{seed}",
        "n_units": n,
        "unit_dim": k,
        "prior": "N(0, I)",
        "note": "Row i is realized unit u_i in one population, not a new population.",
        "representations": _json_float_grid(U),
    }


def _pack_response_law(
    W: np.ndarray,
    E: np.ndarray,
    *,
    sigma: float,
    column_normalize: bool,
) -> dict[str, Any]:
    d, k = int(W.shape[0]), int(W.shape[1])
    n = int(E.shape[0])
    return {
        "form": "unit_specific_factor",
        "assignment": MECHANISM_LAW,
        "note": "Same law for every unit; unit-specificity enters through u_i.",
        "W": {
            "role": "shared_feature_directions",
            "prior": "N(0, I) then optional column L2-normalize",
            "column_normalize": bool(column_normalize),
            "shape": [d, k],
            "values": _json_float_grid(W),
        },
        "noise": {
            "kind": "factual",
            "prior": "N(0, sigma^2)",
            "sigma": float(sigma),
            "independent_of_U": True,
            "shape": [n, d],
            "values": _json_float_grid(E),
        },
    }


def sample_episode(
    rng: np.random.Generator,
    *,
    n_units: int = 64,
    n_features: int = 8,
    unit_dim: int = DEFAULT_UNIT_DIM,
    query_frac: float = DEFAULT_QUERY_FRAC,
    missing_frac: float = DEFAULT_MISSING_FRAC,
    sigma: float = DEFAULT_SIGMA,
    column_normalize: bool = True,
    debug: bool = False,
    return_mechanism: bool = False,
    seed: int | None = 0,
) -> dict[str, Any]:
    n = int(n_units)
    d = int(n_features)
    k = int(unit_dim)
    n_cells = n * d

    U = rng.standard_normal((n, k))
    W = rng.standard_normal((d, k))
    if column_normalize:
        col_norm = np.linalg.norm(W, axis=0, keepdims=True)
        W = W / np.maximum(col_norm, 1e-8)
    E = rng.normal(0.0, float(sigma), size=(n, d))
    Y = U @ W.T + E

    missing_frac = float(np.clip(missing_frac, 0.0, 0.9))
    query_frac = float(np.clip(query_frac, 1e-6, 0.9))
    n_miss = int(round(missing_frac * n_cells))
    n_query = int(round(query_frac * n_cells))
    # At least one context cell and one query cell; missing cannot eat the exam.
    n_miss = max(0, min(n_miss, n_cells - 2))
    n_query = max(1, min(n_query, n_cells - n_miss - 1))

    perm = rng.permutation(n_cells)
    missing_flat = np.zeros(n_cells, dtype=bool)
    query_flat = np.zeros(n_cells, dtype=bool)
    missing_flat[perm[:n_miss]] = True
    query_flat[perm[n_miss : n_miss + n_query]] = True
    missing_mask = missing_flat.reshape(n, d)
    query_mask = query_flat.reshape(n, d)
    context_mask = ~(missing_mask | query_mask)

    values = Y.copy()
    values[query_mask | missing_mask] = np.nan
    y_query = Y[query_mask].astype(np.float64)

    table: dict[str, Any] = {
        "n_units": n,
        "n_features": d,
        "values": _json_float_grid(values),
        "missing_mask": _json_bool_grid(missing_mask),
        "context_mask": _json_bool_grid(context_mask),
        "query_mask": _json_bool_grid(query_mask),
        "y_query": [float(v) for v in y_query],
        "shapes": {
            "values": [n, d],
            "missing_mask": [n, d],
            "context_mask": [n, d],
            "query_mask": [n, d],
            "y_query": [int(query_mask.sum())],
            "n_missing": int(missing_mask.sum()),
            "n_context": int(context_mask.sum()),
            "n_query": int(query_mask.sum()),
        },
    }
    payload: dict[str, Any] = {"seed": seed, "table": table}

    if return_mechanism or debug:
        payload["population"] = _pack_population(U, seed=seed)
        payload["response_law"] = _pack_response_law(
            W, E, sigma=sigma, column_normalize=column_normalize
        )
    if debug:
        payload["Y_full"] = _json_float_grid(Y)
    return payload


def sample_episodes(
    *,
    n_units: int = 64,
    n_features: int = 8,
    unit_dim: int = DEFAULT_UNIT_DIM,
    query_frac: float = DEFAULT_QUERY_FRAC,
    missing_frac: float = DEFAULT_MISSING_FRAC,
    sigma: float = DEFAULT_SIGMA,
    seed: int | None = 0,
    n_episodes: int = 1,
    debug: bool = False,
    return_mechanism: bool = False,
    column_normalize: bool = True,
) -> list[dict[str, Any]]:
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
                missing_frac=missing_frac,
                sigma=sigma,
                column_normalize=column_normalize,
                debug=debug,
                return_mechanism=return_mechanism,
                seed=ep_seed,
            )
        )
    return episodes
