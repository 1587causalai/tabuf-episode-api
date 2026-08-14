"""Pluggable priors that all compile into the same episode table contract.

The shared contract is the WIRE envelope only: a complete table plus
missing_mask and query_mask. Each source has its own row semantics.
DiscoSCM language (units, population, response_law) applies ONLY to
source=discoscm.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from generator import _json_bool_grid, DEFAULT_MISSING_FRAC, DEFAULT_QUERY_FRAC

SKLEARN_SYNTH_CANONICAL = {
    "sklearn_make_classification": "make_classification",
    "sklearn_make_regression": "make_regression",
    "sklearn_friedman1": "make_friedman1",
    "sklearn_low_rank": "make_low_rank_matrix",
}
SKLEARN_SYNTH_MAKER_TO_SOURCE = {v: k for k, v in SKLEARN_SYNTH_CANONICAL.items()}

SKLEARN_REAL_CANONICAL = {
    "sklearn_iris": "iris",
    "sklearn_wine": "wine",
    "sklearn_breast_cancer": "breast_cancer",
    "sklearn_diabetes": "diabetes",
}
SKLEARN_REAL_DS_TO_SOURCE = {v: k for k, v in SKLEARN_REAL_CANONICAL.items()}

SKLEARN_REAL = ("iris", "wine", "breast_cancer", "diabetes")

CANONICAL_SOURCES = (
    "discoscm",
    "scm",
    *SKLEARN_SYNTH_CANONICAL.keys(),
    *SKLEARN_REAL_CANONICAL.keys(),
    "openml",
    "recsys",
)

ALIAS_SOURCES = ("sklearn_synthetic", "sklearn_real")

SOURCES = CANONICAL_SOURCES + ALIAS_SOURCES

_SHARED = (
    "n_units",
    "n_features",
    "seed",
    "n_episodes",
    "batch_size",
    "source",
    "missing_frac",
    "query_frac",
    "query_mode",
    "query_column",
    "return_mechanism",
)
_DISCOSCM_ONLY = ("unit_dim", "type_weights", "independent_frac", "dag_edge_p", "max_parents", "token_heritability", "beta_min", "beta_max", "graph_family")
_SIGMA = ("sigma",)
_DEBUG = ("debug",)
_SOURCE_NAME = ("source_name",)


def _rf(used: tuple[str, ...] | list[str], ignored: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    return {
        "request_fields_used": list(used),
        "request_fields_ignored": list(ignored),
    }


_SKLEARN_USED = _SHARED
_SKLEARN_IGNORED = _DISCOSCM_ONLY + _SIGMA + _DEBUG + _SOURCE_NAME
_ALIAS_USED = _SHARED + _SOURCE_NAME
_ALIAS_IGNORED = _DISCOSCM_ONLY + _SIGMA + _DEBUG
_PLACEHOLDER_USED = _SHARED + _SOURCE_NAME
_PLACEHOLDER_IGNORED = _DISCOSCM_ONLY + _SIGMA + _DEBUG

SOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "discoscm": {
        "status": "ready",
        "family": "discoscm",
        "row_meaning": "unit",
        "query_mode": "cells",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": True,
        **_rf(_SHARED + _DISCOSCM_ONLY + _SIGMA + _DEBUG, _SOURCE_NAME),
        "note": "Rows are units with latent u_i; cell-wise missing/query; unit-specific response law. See data-generation.pdf.",
    },
    "scm": {
        "status": "ready",
        "family": "scm",
        "row_meaning": "iid_sample",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SHARED + _SIGMA, _DISCOSCM_ONLY + _DEBUG + _SOURCE_NAME),
        "note": "i.i.d. rows from a random additive-noise DAG over features; last column is the prediction target.",
    },
    "sklearn_make_classification": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "sklearn make_classification; last column is binary y; no missing by default.",
    },
    "sklearn_make_regression": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "sklearn make_regression; last column is continuous y; no missing by default.",
    },
    "sklearn_friedman1": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "sklearn Friedman #1; last column is continuous y; no missing by default.",
    },
    "sklearn_low_rank": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "cells",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "sklearn low-rank matrix; no designated label; cell-wise missing/query like matrix completion.",
    },
    "sklearn_iris": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "Bundled iris table; last column is the class label.",
    },
    "sklearn_wine": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "Bundled wine table; last column is the class label.",
    },
    "sklearn_breast_cancer": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "Bundled breast_cancer table; last column is the class label.",
    },
    "sklearn_diabetes": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "Bundled diabetes table; last column is the continuous target.",
    },
    "sklearn_synthetic": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_ALIAS_USED, _ALIAS_IGNORED),
        "note": "Alias: pass source_name to pick a maker, or one is drawn at random. Per-maker masks follow the canonical profile.",
        "makers": list(SKLEARN_SYNTH_CANONICAL.values()),
        "alias_of": list(SKLEARN_SYNTH_CANONICAL.keys()),
    },
    "sklearn_real": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_ALIAS_USED, _ALIAS_IGNORED),
        "note": "Alias: pass source_name to pick a bundled table, or one is drawn at random.",
        "datasets": list(SKLEARN_REAL),
        "alias_of": list(SKLEARN_REAL_CANONICAL.keys()),
    },
    "openml": {
        "status": "placeholder",
        "family": "openml",
        "row_meaning": "entity_row",
        "query_mode": "label_column",
        "query_frac": None,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_PLACEHOLDER_USED, _PLACEHOLDER_IGNORED),
        "note": "OpenML classification/regression tables; last column is the label; returns 501 until cached.",
    },
    "recsys": {
        "status": "placeholder",
        "family": "recsys",
        "row_meaning": "user",
        "query_mode": "observed_cells",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_PLACEHOLDER_USED, _PLACEHOLDER_IGNORED),
        "note": "User×Item ratings; unobserved entries are missing; query is a subset of observed ratings; returns 501 until cached.",
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
        raise ValueError(
            "unknown source %r; use %s" % (src, ", ".join(CANONICAL_SOURCES))
        )
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
        base["query_frac"] = DEFAULT_QUERY_FRAC
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
    source: str | None = None,
    mechanism_field: str = "mechanism",
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
        "n_rows": n,
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
        "source": source,
    }
    payload: dict[str, Any] = {"seed": seed, "table": table}
    if return_mechanism and mechanism is not None:
        payload[mechanism_field] = mechanism
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
    source: str | None = None,
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
    source_key = source or SKLEARN_SYNTH_MAKER_TO_SOURCE.get(name, "sklearn_synthetic")
    mech = {"framework": "sklearn_synthetic", "maker": name}
    return pack_grid(
        np.asarray(Y, dtype=np.float64), rng,
        missing_frac=missing_frac, query_frac=query_frac,
        column_types=types, n_classes=n_classes, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode, source=source_key,
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
    source: str | None = None,
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
    source_key = source or SKLEARN_REAL_DS_TO_SOURCE.get(name, "sklearn_real")
    mech = {
        "framework": "sklearn_real",
        "dataset": name,
        "n_rows_original": int(Y0.shape[0]),
        "n_cols_original": int(Y0.shape[1]),
    }
    return pack_grid(
        Y, rng, missing_frac=missing_frac, query_frac=query_frac,
        column_types=types, n_classes=n_classes, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode, source=source_key,
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
    query_mode: str = "label_column",
    source: str | None = None,
) -> dict[str, Any]:
    """Additive-noise SCM: i.i.d. rows from a random DAG over features.

    Last column is the designated prediction target (literature: sample an
    SCM, then pick a node to predict). Rows are observational draws, not units.
    """
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
        "edges": edges,
        "node_fn": node_fn,
        "target_col": d - 1,
    }
    return pack_grid(
        X, rng, missing_frac=missing_frac, query_frac=query_frac,
        column_types=["numeric"] * d, n_classes=[None] * d, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode, source=source or "scm",
    )


def openml_table(*_a, **_k) -> dict[str, Any]:
    raise NotImplementedError(
        "openml source is specified but not wired. Pass a cached dataset name next; default stays discoscm."
    )


def recsys_table(*_a, **_k) -> dict[str, Any]:
    raise NotImplementedError(
        "recsys source is specified (MovieLens / similar) but not cached yet."
    )
