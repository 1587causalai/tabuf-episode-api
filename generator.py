"""Observational v0 TabUF episode generator (DiscoSCM-aligned).

Column types are drawn i.i.d. with weights 70/5/10/5/5
(numeric, ordinal, binary, categorical, high_cardinality), then normalized.
Each column is independently independent-of-rest with probability 0.05.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_UNIT_DIM = 4
DEFAULT_SIGMA = 0.3
DEFAULT_QUERY_FRAC = 0.15
DEFAULT_MISSING_FRAC = 0.05
COL_TYPES = ("numeric", "ordinal", "binary", "categorical", "high_cardinality")
DEFAULT_TYPE_WEIGHTS = {
    "numeric": 70.0,
    "ordinal": 5.0,
    "binary": 10.0,
    "categorical": 5.0,
    "high_cardinality": 5.0,
}
DEFAULT_INDEP_FRAC = 0.05


def _type_probs(weights: dict[str, float] | None) -> np.ndarray:
    src = DEFAULT_TYPE_WEIGHTS if weights is None else weights
    w = np.array([float(src.get(k, 0.0)) for k in COL_TYPES], dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("type_weights must sum to a positive number")
    return w / s


def _json_bool_grid(arr: np.ndarray) -> list[list[bool]]:
    return arr.astype(bool).tolist()


def _json_float_grid(arr: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in arr]


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _n_classes_for(kind: str, n_units: int, rng: np.random.Generator) -> int | None:
    if kind == "numeric":
        return None
    if kind == "binary":
        return 2
    if kind == "ordinal":
        return int(rng.integers(3, 9))
    if kind == "categorical":
        return int(rng.integers(8, 33))
    hi = int(max(48, min(256, max(n_units, 48))))
    lo = int(max(32, min(64, hi - 1)))
    return int(rng.integers(lo, hi + 1))


def _realize_column(
    rng: np.random.Generator,
    *,
    kind: str,
    independent: bool,
    n_classes: int | None,
    U: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unit-specific response for one feature.

    numeric:    Y = scale * <u, w> + shift + e
    ordinal:    ordered probit, cuts on <u, w> + e
    categorical / high_cardinality: multinomial logit, Y ~ softmax(<u, C_c>)
    independent: drop u, draw from a column-wise marginal of the same type.
    """
    n, k = int(U.shape[0]), int(U.shape[1])
    meta: dict[str, Any] = {
        "type": kind,
        "independent": bool(independent),
        "n_classes": n_classes,
    }

    if kind == "numeric":
        if independent:
            mu = float(rng.normal(0.0, 1.0))
            sig = float(np.exp(rng.normal(-0.2, 0.4)))
            y = rng.normal(mu, sig, size=n)
            meta["g"] = "independent_gaussian"
            meta["mu"] = mu
            meta["sigma"] = sig
            return y.astype(np.float64), meta
        w = rng.standard_normal(k)
        w = w / max(float(np.linalg.norm(w)), 1e-8)
        scale = float(np.exp(rng.normal(0.0, 0.35)))
        shift = float(rng.normal(0.0, 0.5))
        e = rng.normal(0.0, float(sigma), size=n)
        y = scale * (U @ w) + shift + e
        meta["g"] = "affine_factor"
        meta["w"] = [float(x) for x in w]
        meta["scale"] = scale
        meta["shift"] = shift
        meta["sigma"] = float(sigma)
        return y.astype(np.float64), meta

    if kind == "binary":
        if independent:
            p = float(rng.uniform(0.1, 0.9))
            y = rng.binomial(1, p, size=n)
            meta["g"] = "independent_bernoulli"
            meta["p"] = p
            return y.astype(np.int64), meta
        w = rng.standard_normal(k)
        w = w / max(float(np.linalg.norm(w)), 1e-8)
        e = rng.normal(0.0, float(sigma), size=n)
        y = (U @ w + e > 0.0).astype(np.int64)
        meta["g"] = "latent_threshold"
        meta["w"] = [float(x) for x in w]
        meta["sigma"] = float(sigma)
        return y, meta

    assert n_classes is not None
    L = int(n_classes)

    if independent:
        alpha = np.ones(L) * (0.4 if kind == "high_cardinality" else 0.9)
        p = rng.dirichlet(alpha)
        y = rng.choice(L, size=n, p=p)
        meta["g"] = "independent_multinomial"
        meta["probs"] = [float(x) for x in p]
        return y.astype(np.int64), meta

    if kind == "ordinal":
        w = rng.standard_normal(k)
        w = w / max(float(np.linalg.norm(w)), 1e-8)
        e = rng.normal(0.0, float(sigma), size=n)
        s = U @ w + e
        cuts = np.sort(rng.normal(0.0, 1.0, size=L - 1))
        y = np.sum(s[:, None] > cuts[None, :], axis=1)
        meta["g"] = "ordered_probit"
        meta["w"] = [float(x) for x in w]
        meta["thresholds"] = [float(x) for x in cuts]
        meta["sigma"] = float(sigma)
        return y.astype(np.int64), meta

    # Unordered: class embeddings C (L x k). High-card scales are heterogeneous
    # so a few classes dominate (Zipf-like), many stay rare.
    C = rng.standard_normal((L, k))
    if kind == "high_cardinality":
        scales = np.exp(rng.normal(0.0, 0.9, size=L))
        C = C * scales[:, None]
        meta["class_scales"] = [float(x) for x in scales]
    logits = U @ C.T
    gumbel = rng.gumbel(size=(n, L))
    y = np.argmax(logits + gumbel, axis=1)
    meta["g"] = "multinomial_logit_gumbel"
    meta["class_embeddings"] = _json_float_grid(C)
    return y.astype(np.int64), meta


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
    type_weights: dict[str, float] | None = None,
    independent_frac: float = DEFAULT_INDEP_FRAC,
) -> dict[str, Any]:
    n = int(n_units)
    d = int(n_features)
    k = int(unit_dim)
    n_cells = n * d
    probs = _type_probs(type_weights)
    indep_p = float(np.clip(independent_frac, 0.0, 1.0))
    used_weights = {
        k: float((type_weights or DEFAULT_TYPE_WEIGHTS).get(k, DEFAULT_TYPE_WEIGHTS[k]))
        for k in COL_TYPES
    }

    U = rng.standard_normal((n, k))
    kinds = rng.choice(COL_TYPES, size=d, p=probs)
    independent = rng.random(d) < indep_p
    n_classes = [_n_classes_for(str(kinds[j]), n, rng) for j in range(d)]

    Y_cols: list[np.ndarray] = []
    col_maps: list[dict[str, Any]] = []
    for j in range(d):
        yj, meta = _realize_column(
            rng,
            kind=str(kinds[j]),
            independent=bool(independent[j]),
            n_classes=n_classes[j],
            U=U,
            sigma=sigma,
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
    n_miss = max(0, min(int(round(missing_frac * n_cells)), n_cells))
    n_query = max(1, min(int(round(query_frac * n_cells)), n_cells))
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
            "form": "per_column_unit_specific",
            "type_weights": used_weights,
            "column_independent_prob": indep_p,
            "columns": col_maps,
        }
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
    type_weights: dict[str, float] | None = None,
    independent_frac: float = DEFAULT_INDEP_FRAC,
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
                type_weights=type_weights,
                independent_frac=independent_frac,
            )
        )
    return episodes
