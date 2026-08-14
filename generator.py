"""Observational v0 TabUF episode generator (DiscoSCM-aligned).

Each response is n episodes; each episode samples its own population.
The table is complete. missing_mask and query_mask are independent and may
overlap: query ∩ missing is imputation, query without missing is ordinary prediction.
No separate y_query. Column types are mixed by default.
"""

from __future__ import annotations

from math import erf
from typing import Any

import numpy as np

DEFAULT_UNIT_DIM = 4
DEFAULT_SIGMA = 0.3
DEFAULT_QUERY_FRAC = 0.15
DEFAULT_MISSING_FRAC = 0.05
MECHANISM_LAW = "s[i,j] = <U[i], W[j]> + E[i,j]; Y[i,j] = g_j(s[i,j])"
COL_TYPES = ("numeric", "ordinal", "categorical", "high_cardinality")
COL_PROBS = np.array([0.40, 0.20, 0.25, 0.15])
INDEP_P = 0.10


def _json_bool_grid(arr: np.ndarray) -> list[list[bool]]:
    return arr.astype(bool).tolist()


def _json_float_grid(arr: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in arr]


def _phi(s: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.sqrt(2.0)
    s = np.asarray(s, dtype=np.float64)
    out = np.empty_like(s)
    for idx, v in enumerate(s.ravel()):
        out.ravel()[idx] = 0.5 * (1.0 + erf(float(v) * inv))
    return out


def _n_classes_for(kind: str, n_units: int, rng: np.random.Generator) -> int | None:
    if kind == "numeric":
        return None
    if kind == "ordinal":
        return int(rng.integers(3, 9))
    if kind == "categorical":
        return int(rng.integers(8, 33))
    hi = int(max(48, min(256, max(n_units, 48))))
    lo = int(max(32, min(64, hi - 1)))
    return int(rng.integers(lo, hi + 1))


def _bin_latent(s: np.ndarray, n_classes: int) -> np.ndarray:
    codes = np.floor(_phi(s) * n_classes).astype(np.int64)
    return np.clip(codes, 0, n_classes - 1)


def _realize_column(
    rng: np.random.Generator,
    *,
    kind: str,
    independent: bool,
    n_classes: int | None,
    s: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return length-n values and the column map stored in the response law."""
    n = int(s.shape[0])
    meta: dict[str, Any] = {"type": kind, "independent": bool(independent), "n_classes": n_classes}
    if kind == "numeric":
        if independent:
            y = rng.standard_normal(n)
            meta["marginal"] = "N(0,1)"
        else:
            y = s.astype(np.float64)
            meta["g"] = "identity"
        return y, meta

    assert n_classes is not None
    if independent:
        y = rng.integers(0, n_classes, size=n)
        meta["marginal"] = f"Unif{{0,...,{n_classes - 1}}}"
        return y.astype(np.int64), meta

    codes = _bin_latent(s, n_classes)
    if kind == "ordinal":
        meta["g"] = "ordered_probit_bins"
        return codes, meta
    perm = rng.permutation(n_classes)
    meta["g"] = "binned_then_label_perm"
    meta["label_perm"] = [int(x) for x in perm]
    return perm[codes], meta


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
    S = U @ W.T + E

    kinds = rng.choice(COL_TYPES, size=d, p=COL_PROBS)
    independent = rng.random(d) < INDEP_P
    n_classes = [_n_classes_for(str(kinds[j]), n, rng) for j in range(d)]

    Y_cols: list[np.ndarray] = []
    col_maps: list[dict[str, Any]] = []
    for j in range(d):
        yj, meta = _realize_column(
            rng,
            kind=str(kinds[j]),
            independent=bool(independent[j]),
            n_classes=n_classes[j],
            s=S[:, j],
        )
        meta["j"] = j
        Y_cols.append(yj)
        col_maps.append(meta)

    values: list[list[float | int]] = []
    for i in range(n):
        row: list[float | int] = []
        for j in range(d):
            v = Y_cols[j][i]
            row.append(int(v) if n_classes[j] is not None else float(v))
        values.append(row)

    missing_frac = float(np.clip(missing_frac, 0.0, 0.95))
    query_frac = float(np.clip(query_frac, 0.0, 0.95))
    n_miss = int(round(missing_frac * n_cells))
    n_query = int(round(query_frac * n_cells))
    n_miss = max(0, min(n_miss, n_cells))
    n_query = max(1, min(n_query, n_cells))
    missing_flat = np.zeros(n_cells, dtype=bool)
    query_flat = np.zeros(n_cells, dtype=bool)
    if n_miss:
        missing_flat[rng.choice(n_cells, size=n_miss, replace=False)] = True
    query_flat[rng.choice(n_cells, size=n_query, replace=False)] = True
    missing_mask = missing_flat.reshape(n, d)
    query_mask = query_flat.reshape(n, d)

    table: dict[str, Any] = {
        "n_units": n,
        "n_features": d,
        "values": values,
        "missing_mask": _json_bool_grid(missing_mask),
        "query_mask": _json_bool_grid(query_mask),
        "column_types": [str(x) for x in kinds],
        "n_classes": n_classes,
        "shapes": {
            "values": [n, d],
            "missing_mask": [n, d],
            "query_mask": [n, d],
            "n_missing": int(missing_mask.sum()),
            "n_query": int(query_mask.sum()),
            "n_query_and_missing": int((query_mask & missing_mask).sum()),
        },
    }
    payload: dict[str, Any] = {"seed": seed, "table": table}

    if return_mechanism or debug:
        payload["population"] = {
            "id": f"pop-{seed}",
            "n_units": n,
            "unit_dim": k,
            "prior": "N(0, I)",
            "note": "This episode sampled its own population. Row i is unit u_i.",
            "representations": _json_float_grid(U),
        }
        payload["response_law"] = {
            "form": "unit_specific_then_column_map",
            "assignment": MECHANISM_LAW,
            "column_independent_prob": INDEP_P,
            "W": {
                "role": "shared_feature_directions",
                "column_normalize": bool(column_normalize),
                "shape": [d, k],
                "values": _json_float_grid(W),
            },
            "noise": {
                "kind": "factual",
                "sigma": float(sigma),
                "independent_of_U": True,
                "shape": [n, d],
                "values": _json_float_grid(E),
            },
            "columns": col_maps,
        }
    if debug:
        payload["S"] = _json_float_grid(S)
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
