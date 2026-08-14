"""Pluggable priors that all compile into the same episode table contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from generator import _json_bool_grid, DEFAULT_MISSING_FRAC, DEFAULT_QUERY_FRAC

SOURCES = (
    "discoscm",
    "sklearn_synthetic",
    "sklearn_real",
    "scm",
    "openml",
    "recsys",
)

SKLEARN_REAL = ("iris", "wine", "breast_cancer", "diabetes")

# Per-source default compiler policy. query_mode:
#   cells          — random cells (DiscoSCM / feature SCM)
#   label_column   — last column is the task label; query that whole column
#   observed_cells — recsys: query a subset of observed ratings (placeholder)
SOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "discoscm": {
        "status": "ready",
        "task": "grid",
        "query_mode": "cells",
        "query_frac": 0.15,
        "missing_frac": 0.05,
        "note": "Unit-specific response law. Default cell-wise C/Q.",
    },
    "sklearn_synthetic": {
        "status": "ready",
        "task": "supervised",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "note": "Last column is y. Default query is the label column.",
        "makers": ["make_classification", "make_regression", "make_friedman1", "make_low_rank_matrix"],
    },
    "sklearn_real": {
        "status": "ready",
        "task": "supervised",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "note": "Bundled sklearn tables. Default query is the label column.",
        "datasets": list(SKLEARN_REAL),
    },
    "scm": {
        "status": "ready",
        "task": "grid",
        "query_mode": "cells",
        "query_frac": 0.15,
        "missing_frac": 0.05,
        "note": "Paper-style feature ANM. Rows i.i.d. Default cell-wise masks.",
    },
    "openml": {
        "status": "placeholder",
        "task": "supervised",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "note": "OpenML clf/reg. Same contract as sklearn_real once cached.",
    },
    "recsys": {
        "status": "placeholder",
        "task": "ratings",
        "query_mode": "observed_cells",
        "query_frac": 0.15,
        "missing_frac": 0.0,
        "note": "User x item ratings. Query a subset of observed entries; unobserved stay missing.",
    },
}


def resolve_profile(
    source: str,
    *,
    query_mode: str | None = None,
    missing_frac: float | None = None,
    query_frac: float | None = None,
) -> dict[str, Any]:
    src = (source or "discoscm").lower()
    if src not in SOURCE_PROFILES:
        raise ValueError("unknown source %r" % src)
    base = dict(SOURCE_PROFILES[src])
    if query_mode:
        base["query_mode"] = query_mode
    if missing_frac is not None:
        base["missing_frac"] = missing_frac
    if query_frac is not None:
        base["query_frac"] = query_frac
    if base.get("missing_frac") is None:
        base["missing_frac"] = 0.0
    if base.get("query_frac") is None:
        base["query_frac"] = 0.15
    return base


def _masks(
    rng: np.random.Generator,
    n: int,
    d: int,
    missing_frac: float,
    query_frac: float,
    query_mode: str,
):
    n_cells = n * d
    n_miss = max(0, min(int(round(float(missing_frac) * n_cells)), n_cells))
    missing_flat = np.zeros(n_cells, dtype=bool)
    if n_miss:
        missing_flat[rng.choice(n_cells, size=n_miss, replace=False)] = True
    missing_mask = missing_flat.reshape(n, d)
    query_mask = np.zeros((n, d), dtype=bool)
    mode = query_mode or "cells"
    if mode == "label_column":
        query_mask[:, -1] = True
    elif mode == "observed_cells":
        # Placeholder behaviour: same as cells, until recsys cache exists.
        n_query = max(1, min(int(round(float(query_frac) * n_cells)), n_cells))
        query_flat = np.zeros(n_cells, dtype=bool)
        query_flat[rng.choice(n_cells, size=n_query, replace=False)] = True
        query_mask = query_flat.reshape(n, d)
    else:
        n_query = max(1, min(int(round(float(query_frac) * n_cells)), n_cells))
        query_flat = np.zeros(n_cells, dtype=bool)
        query_flat[rng.choice(n_cells, size=n_query, replace=False)] = True
        query_mask = query_flat.reshape(n, d)
    return missing_mask, query_mask


def pack_grid(
    Y: np.ndarray,
    rng: np.random.Generator,
    *,
    missing_frac: float,
    query_frac: float,
    column_types: list[str],
    n_classes: list[int | None],
    seed: int | None,
    return_mechanism: bool,
    mechanism: dict[str, Any] | None,
    query_mode: str = "cells",
) -> dict[str, Any]:
    n, d = int(Y.shape[0]), int(Y.shape[1])
    missing_mask, query_mask = _masks(rng, n, d, missing_frac, query_frac, query_mode)
    values: list[list[float | int]] = []
    for i in range(n):
        row: list[float | int] = []
        for j in range(d):
            v = Y[i, j]
            row.append(int(v) if n_classes[j] is not None else float(v))
        values.append(row)
    table = {
        "n_units": n,
        "n_features": d,
        "values": values,
        "missing_mask": _json_bool_grid(missing_mask),
        "query_mask": _json_bool_grid(query_mask),
        "column_types": column_types,
        "n_classes": n_classes,
        "shapes": {
            "values": [n, d],
            "missing_mask": [n, d],
            "query_mask": [n, d],
            "n_missing": int(missing_mask.sum()),
            "n_query": int(query_mask.sum()),
            "n_query_and_missing": int((query_mask & missing_mask).sum()),
        },
        "query_mode": query_mode,
    }
    payload: dict[str, Any] = {"seed": seed, "table": table}
    if return_mechanism and mechanism is not None:
        payload["response_law"] = mechanism
    return payload


def _fit_size(X: np.ndarray, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Subsample or pad-with-repeat to requested n x d."""
    r, c = X.shape
    rows = rng.choice(r, size=n, replace=(r < n))
    cols = rng.choice(c, size=d, replace=(c < d))
    return np.asarray(X[np.ix_(rows, cols)], dtype=np.float64)


def _fit_supervised(Y: np.ndarray, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Keep the last column as label; subsample rows and the other features."""
    r, c = Y.shape
    rows = rng.choice(r, size=n, replace=(r < n))
    ylab = Y[rows, -1]
    if d <= 1:
        return ylab.reshape(-1, 1)
    feat = max(c - 1, 1)
    need = d - 1
    cols = rng.choice(feat, size=need, replace=(feat < need))
    return np.column_stack([Y[rows][:, cols], ylab])


def sklearn_synthetic(
    rng: np.random.Generator,
    *,
    n_units: int,
    n_features: int,
    missing_frac: float,
    query_frac: float,
    seed: int | None,
    return_mechanism: bool,
    source_name: str | None = None,
    query_mode: str = "label_column",
) -> dict[str, Any]:
    from sklearn.datasets import (
        make_classification,
        make_friedman1,
        make_low_rank_matrix,
        make_regression,
    )

    makers = {
        "make_classification": make_classification,
        "make_regression": make_regression,
        "make_friedman1": make_friedman1,
        "make_low_rank_matrix": make_low_rank_matrix,
    }
    name = source_name or str(rng.choice(list(makers)))
    if name not in makers:
        raise ValueError(f"unknown sklearn synthetic maker: {name}")
    rs = int(seed or 0)
    n, d = int(n_units), int(n_features)
    if name == "make_classification":
        X, y = make_classification(
            n_samples=n, n_features=max(d - 1, 2), n_informative=max(d // 2, 1),
            n_redundant=0, n_classes=2, random_state=rs,
        )
        if d >= 2:
            Y = np.column_stack([X[:, : d - 1], y])
        else:
            Y = y.reshape(-1, 1).astype(np.float64)
        types = ["numeric"] * (d - 1) + ["binary"] if d >= 2 else ["binary"]
        n_classes = [None] * (d - 1) + [2] if d >= 2 else [2]
    elif name == "make_regression":
        X, y = make_regression(n_samples=n, n_features=max(d - 1, 1), noise=0.3, random_state=rs)
        Y = np.column_stack([X[:, : max(d - 1, 1)], y])[:, :d]
        types = ["numeric"] * d
        n_classes = [None] * d
    elif name == "make_friedman1":
        nf = max(5, d)
        X, y = make_friedman1(n_samples=n, n_features=nf, noise=0.3, random_state=rs)
        Y = _fit_supervised(np.column_stack([X, y]), n, d, rng)
        types = ["numeric"] * d
        n_classes = [None] * d
    else:
        Y = make_low_rank_matrix(n_samples=n, n_features=d, random_state=rs)
        types = ["numeric"] * d
        n_classes = [None] * d
    mech = {"framework": "sklearn_synthetic", "maker": name, "note": "Rows are i.i.d. draws, not DiscoSCM units."}
    return pack_grid(
        np.asarray(Y, dtype=np.float64), rng,
        missing_frac=missing_frac, query_frac=query_frac,
        column_types=types, n_classes=n_classes, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode,
    )


def sklearn_real(
    rng: np.random.Generator,
    *,
    n_units: int,
    n_features: int,
    missing_frac: float,
    query_frac: float,
    seed: int | None,
    return_mechanism: bool,
    source_name: str | None = None,
    query_mode: str = "label_column",
) -> dict[str, Any]:
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "diabetes": load_diabetes,
    }
    name = source_name or str(rng.choice(list(loaders)))
    if name not in loaders:
        raise ValueError(f"unknown sklearn real dataset: {name}. try {SKLEARN_REAL}")
    bunch = loaders[name]()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target)
    Y0 = np.column_stack([X, y.reshape(-1, 1)])
    Y = _fit_supervised(Y0, n_units, n_features, rng)
    types = ["numeric"] * n_features
    n_classes: list[int | None] = [None] * n_features
    mech = {
        "framework": "sklearn_real",
        "dataset": name,
        "n_rows_original": int(Y0.shape[0]),
        "n_cols_original": int(Y0.shape[1]),
        "note": "Real table subsampled into Unit x Feature grid. No unit-specific law.",
    }
    return pack_grid(
        Y, rng, missing_frac=missing_frac, query_frac=query_frac,
        column_types=types, n_classes=n_classes, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode,
    )


def scm_anm(
    rng: np.random.Generator,
    *,
    n_units: int,
    n_features: int,
    missing_frac: float,
    query_frac: float,
    seed: int | None,
    return_mechanism: bool,
    sigma: float,
    query_mode: str = "cells",
) -> dict[str, Any]:
    """Paper-style additive-noise SCM: i.i.d. rows from a random DAG."""
    n, d = int(n_units), int(n_features)
    X = np.zeros((n, d), dtype=np.float64)
    edges: list[list[int]] = []
    node_fn: list[str] = []
    for j in range(d):
        e = rng.normal(0.0, float(sigma), size=n)
        if j == 0:
            X[:, j] = e
            node_fn.append("root_noise")
            continue
        n_pa = int(rng.integers(0, min(3, j) + 1))
        pa = rng.choice(j, size=n_pa, replace=False).tolist() if n_pa else []
        if not pa:
            X[:, j] = e
            node_fn.append("root_noise")
            continue
        w = rng.normal(0.0, 1.0, size=n_pa)
        lin = X[:, np.array(pa, dtype=int)] @ w
        kind = "linear" if rng.random() < 0.5 else "tanh"
        X[:, j] = (lin if kind == "linear" else np.tanh(lin)) + e
        node_fn.append(kind)
        for p in pa:
            edges.append([int(p), j])
    mech = {
        "framework": "ANM-SCM",
        "note": "TabPFN/TabICL-style feature SCM. Rows are i.i.d.; nodes are features, not units.",
        "edges": edges,
        "node_fn": node_fn,
    }
    return pack_grid(
        X, rng, missing_frac=missing_frac, query_frac=query_frac,
        column_types=["numeric"] * d, n_classes=[None] * d, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode,
    )


def openml_table(*_a, **_k) -> dict[str, Any]:
    raise NotImplementedError(
        "openml source is specified but not wired. Pass a cached dataset name next; default stays discoscm."
    )


def recsys_table(*_a, **_k) -> dict[str, Any]:
    raise NotImplementedError(
        "recsys source is specified (MovieLens / similar) but not cached yet."
    )
