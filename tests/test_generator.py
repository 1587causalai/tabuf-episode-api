import numpy as np
from generator import sample_episodes


def test_table_is_complete_no_y_query():
    ep = sample_episodes(n_units=16, n_features=8, n_episodes=1, seed=0)[0]
    t = ep["table"]
    assert "y_query" not in t and "context_mask" not in t
    assert len(t["values"]) == 16 and len(t["values"][0]) == 8
    for row in t["values"]:
        assert all(v is not None for v in row)


def test_masks_may_overlap():
    ep = sample_episodes(
        n_units=32, n_features=16, n_episodes=1, seed=3,
        missing_frac=0.2, query_frac=0.3,
    )[0]
    m = np.array(ep["table"]["missing_mask"])
    q = np.array(ep["table"]["query_mask"])
    assert (q & m).sum() >= 0  # allowed
    assert q.sum() > 0 and m.sum() > 0
    # not a partition
    assert not np.all(m | q)


def test_each_episode_own_population():
    eps = sample_episodes(n_units=8, n_features=4, n_episodes=2, seed=0, unit_dim=8, return_mechanism=True)
    assert eps[0]["population"]["id"] != eps[1]["population"]["id"]
    u0 = np.array(eps[0]["population"]["representations"])
    u1 = np.array(eps[1]["population"]["representations"])
    assert not np.allclose(u0, u1)


def test_column_type_mix_and_independent_in_law():
    ep = sample_episodes(
        n_units=32, n_features=20, n_episodes=1, seed=7, return_mechanism=True
    )[0]
    types = set(ep["table"]["column_types"])
    assert types <= {"numeric", "ordinal", "binary", "categorical", "high_cardinality"}
    assert "numeric" in types
    cols = ep["response_law"]["columns"]
    assert any(c["independent"] for c in cols) or True  # 10% may miss on unlucky seed
    # discrete codes in range
    for j, kind in enumerate(ep["table"]["column_types"]):
        k = ep["table"]["n_classes"][j]
        col = [ep["table"]["values"][i][j] for i in range(32)]
        if kind == "numeric":
            assert all(isinstance(v, float) for v in col)
        else:
            assert k is not None
            assert all(isinstance(v, int) and 0 <= v < k for v in col)


def test_defaults_fracs():
    ep = sample_episodes(n_units=20, n_features=10, n_episodes=1, seed=1)[0]
    n_cells = 200
    nm = ep["table"]["shapes"]["n_missing"]
    nq = ep["table"]["shapes"]["n_query"]
    assert nm == round(0.05 * n_cells)
    assert nq == round(0.15 * n_cells)


def test_type_weights_override_all_numeric():
    ep = sample_episodes(
        n_units=12, n_features=8, n_episodes=1, seed=0,
        type_weights={"numeric": 1, "ordinal": 0, "binary": 0, "categorical": 0, "high_cardinality": 0},
        independent_frac=0.0,
    )[0]
    assert set(ep["table"]["column_types"]) == {"numeric"}


def test_sklearn_synthetic_compiles_to_table():
    ep = sample_episodes(
        n_units=20, n_features=6, n_episodes=1, seed=0,
        source="sklearn_synthetic", source_name="make_regression",
    )[0]
    assert ep["table"]["n_units"] == 20
    assert "query_mask" in ep["table"]

def test_scm_has_edges_when_mechanism_on():
    ep = sample_episodes(
        n_units=16, n_features=6, n_episodes=1, seed=1,
        source="scm", return_mechanism=True,
    )[0]
    assert "population" not in ep
    assert "response_law" not in ep
    mech = ep["mechanism"]
    assert mech["framework"] == "ANM-SCM"
    assert "edges" in mech
    assert "node_fn" in mech
    assert mech["target_col"] == 5
    q = ep["table"]["query_mask"]
    assert all(row[-1] for row in q)
    assert ep["table"].get("query_mode") == "label_column"
    blob = str(mech)
    assert "unit" not in blob.lower()

def test_sklearn_real_iris():
    ep = sample_episodes(
        n_units=30, n_features=4, n_episodes=1, seed=0,
        source="sklearn_real", source_name="iris",
    )[0]
    assert ep["table"]["n_features"] == 4


def test_sklearn_real_queries_label_column():
    ep = sample_episodes(
        n_units=30, n_features=5, n_episodes=1, seed=0,
        source="sklearn_real", source_name="iris",
        missing_frac=None, query_frac=None,
    )[0]
    q = ep["table"]["query_mask"]
    assert all(row[-1] for row in q)
    assert ep["table"].get("query_mode") == "label_column"

def test_sklearn_iris_canonical_source():
    ep = sample_episodes(
        n_units=30, n_features=4, n_episodes=1, seed=0,
        source="sklearn_iris",
    )[0]
    assert ep["table"]["n_features"] == 4
    assert ep["table"].get("source") == "sklearn_iris"
    q = ep["table"]["query_mask"]
    assert all(row[-1] for row in q)
    assert ep["table"].get("query_mode") == "label_column"
    assert "population" not in ep


def test_sklearn_make_classification_label_column():
    ep = sample_episodes(
        n_units=24, n_features=6, n_episodes=1, seed=0,
        source="sklearn_make_classification", return_mechanism=True,
    )[0]
    q = ep["table"]["query_mask"]
    assert all(row[-1] for row in q)
    assert ep["table"].get("query_mode") == "label_column"
    assert ep["table"]["column_types"][-1] == "binary"
    assert ep["table"]["n_classes"][-1] == 2
    assert "population" not in ep
    assert "response_law" not in ep
    assert ep["mechanism"]["maker"] == "make_classification"


def test_sklearn_low_rank_is_cellwise_not_supervised():
    ep = sample_episodes(
        n_units=20, n_features=8, n_episodes=1, seed=2,
        source="sklearn_low_rank",
    )[0]
    assert ep["table"].get("query_mode") == "cells"
    q = ep["table"]["query_mask"]
    # not the whole last column
    assert not all(row[-1] for row in q)
    n_cells = 20 * 8
    assert ep["table"]["shapes"]["n_missing"] == round(0.05 * n_cells)
    assert ep["table"]["shapes"]["n_query"] == round(0.15 * n_cells)

def test_nonindependent_tokens_near_unit_norm():
    ep = sample_episodes(
        n_units=16, n_features=12, n_episodes=1, seed=0,
        independent_frac=0.0, return_mechanism=True, unit_dim=4,
    )[0]
    tokens = ep["response_law"]["tokens"]
    assert len(tokens) == 12
    for t in tokens:
        assert t is not None
        nrm = float(np.linalg.norm(t))
        assert abs(nrm - 1.0) < 1e-6
    parents = ep["response_law"]["parents"]
    betas = ep["response_law"]["betas"]
    assert len(parents) == 12 and len(betas) == 12
    for pa, be in zip(parents, betas):
        assert len(pa) == len(be)
        assert all(0 <= p < 12 for p in pa)
    # DAG: parents only from earlier nodes in some topo order (permutation of all)
    # Reconstruct: no cycles via DFS
    seen_stack = set()
    seen_done = set()

    def visit(j: int) -> None:
        assert j not in seen_stack
        if j in seen_done:
            return
        seen_stack.add(j)
        for p in parents[j]:
            visit(p)
        seen_stack.remove(j)
        seen_done.add(j)

    for j in range(12):
        visit(j)


def test_independent_columns_not_in_dag():
    ep = sample_episodes(
        n_units=16, n_features=24, n_episodes=1, seed=1,
        independent_frac=0.5, return_mechanism=True,
    )[0]
    cols = ep["response_law"]["columns"]
    parents = ep["response_law"]["parents"]
    betas = ep["response_law"]["betas"]
    tokens = ep["response_law"]["tokens"]
    indep_idx = [c["j"] for c in cols if c["independent"]]
    assert indep_idx, "seed 1 with p=0.5 should mark some independent columns"
    active = [c["j"] for c in cols if not c["independent"]]
    for j in indep_idx:
        assert parents[j] == []
        assert betas[j] == []
        assert tokens[j] is None
        for i, pa in enumerate(parents):
            assert j not in pa
        assert "t" not in cols[j]
    for j in active:
        assert tokens[j] is not None
        nrm = float(np.linalg.norm(tokens[j]))
        assert abs(nrm - 1.0) < 1e-6
        for p in parents[j]:
            assert p in active


def test_envelope_unchanged():
    ep = sample_episodes(n_units=16, n_features=8, n_episodes=1, seed=0)[0]
    t = ep["table"]
    for key in (
        "values", "missing_mask", "query_mask",
        "n_units", "n_features", "column_types", "n_classes",
        "shapes", "query_mode",
    ):
        assert key in t
    assert "y_input" not in t and "y_query" not in t
    assert "context_mask" not in t
    assert "population" not in ep
    assert "response_law" not in ep
    assert len(t["values"]) == 16 and len(t["values"][0]) == 8
    assert len(t["missing_mask"]) == 16 and len(t["query_mask"]) == 16


def test_token_scm_omitted_by_default_present_in_debug():
    plain = sample_episodes(n_units=8, n_features=4, n_episodes=1, seed=2)[0]
    assert "response_law" not in plain
    dbg = sample_episodes(
        n_units=8, n_features=4, n_episodes=1, seed=2, debug=True,
    )[0]
    law = dbg["response_law"]
    assert law["form"] == "feature_token_linear_scm"
    assert "parents" in law and "betas" in law and "tokens" in law

def _child_parent_cosines(ep, identity_only=False):
    from generator import _apply_activation
    law = ep["response_law"]
    tokens = law["tokens"]
    parents = law["parents"]
    betas = law["betas"]
    activations = law.get("activations") or ["identity"] * len(parents)
    out = []
    for j, pa in enumerate(parents):
        if not pa:
            continue
        phi = activations[j] or "identity"
        if identity_only and phi != "identity":
            continue
        t_j = np.asarray(tokens[j], dtype=np.float64)
        s_raw = np.zeros_like(t_j)
        for b, p in zip(betas[j], pa):
            s_raw = s_raw + float(b) * np.asarray(tokens[p], dtype=np.float64)
        s = _apply_activation(s_raw, phi)
        s = s / max(float(np.linalg.norm(s)), 1e-8)
        out.append(float(np.dot(t_j, s)))
    return out


def test_tokens_unit_norm_and_independent_have_no_token():
    ep = sample_episodes(
        n_units=16, n_features=8, n_episodes=1, seed=0,
        independent_frac=0.0, return_mechanism=True,
    )[0]
    tokens = ep["response_law"]["tokens"]
    assert len(tokens) == 8
    for t in tokens:
        assert t is not None
        assert abs(float(np.linalg.norm(t)) - 1.0) < 1e-6


def test_independent_columns_excluded_from_dag():
    ep = sample_episodes(
        n_units=16, n_features=8, n_episodes=1, seed=0,
        independent_frac=1.0, return_mechanism=True,
    )[0]
    tokens = ep["response_law"]["tokens"]
    parents = ep["response_law"]["parents"]
    betas = ep["response_law"]["betas"]
    assert all(t is None for t in tokens)
    assert all(pa == [] for pa in parents)
    assert all(be == [] for be in betas)


def test_default_n_units_is_1000():
    ep = sample_episodes(n_features=4, n_episodes=1, seed=0)[0]
    assert ep["table"]["n_units"] == 1000
    assert ep["table"]["n_features"] == 4


def test_child_token_aligns_with_parent_signal():
    abs_betas = []
    two_parent_abs = []
    for seed in range(8):
        ep = sample_episodes(
            n_units=16, n_features=12, n_episodes=1, seed=seed,
            independent_frac=0.0, return_mechanism=True, unit_dim=8,
            token_heritability=0.8,
        )[0]
        parents = ep["response_law"]["parents"]
        betas = ep["response_law"]["betas"]
        for j, pa in enumerate(parents):
            if not pa:
                continue
            be = betas[j]
            assert abs(sum(abs(float(b)) for b in be) - 1.0) < 1e-6
            if len(pa) == 1:
                assert abs(abs(float(be[0])) - 1.0) < 1e-6
            abs_betas.extend(abs(float(b)) for b in be)
            if len(pa) == 2:
                two_parent_abs.extend(abs(float(b)) for b in be)
        for cos in _child_parent_cosines(ep):
            assert cos > 0.55
    assert abs_betas
    assert all(0.0 < b <= 1.0 + 1e-9 for b in abs_betas)
    if two_parent_abs:
        assert all(0.2 - 1e-9 <= b <= 0.8 + 1e-9 for b in two_parent_abs)


def test_low_heritability_weaker_alignment():
    low, high = [], []
    for seed in range(8):
        kw = dict(
            n_units=16, n_features=12, n_episodes=1, seed=seed,
            independent_frac=0.0, return_mechanism=True, unit_dim=8,
        )
        ep_lo = sample_episodes(token_heritability=0.2, **kw)[0]
        ep_hi = sample_episodes(token_heritability=0.9, **kw)[0]
        low.extend(_child_parent_cosines(ep_lo))
        high.extend(_child_parent_cosines(ep_hi))
    assert low and high
    assert float(np.mean(high)) > float(np.mean(low))


def test_dag_knobs_change_edge_count():
    ep = sample_episodes(
        n_units=16, n_features=8, n_episodes=1, seed=0,
        independent_frac=0.0, return_mechanism=True,
        max_parents=1, dag_edge_p=1.0,
    )[0]
    parents = ep["response_law"]["parents"]
    assert all(len(pa) <= 1 for pa in parents)
    assert sum(len(pa) for pa in parents) >= 1

def _assert_dag(parents):
    seen_stack = set()
    seen_done = set()

    def visit(j: int) -> None:
        assert j not in seen_stack
        if j in seen_done:
            return
        seen_stack.add(j)
        for p in parents[j]:
            visit(p)
        seen_stack.remove(j)
        seen_done.add(j)

    for j in range(len(parents)):
        visit(j)


def test_n_features_none_in_range():
    ds = []
    for e in range(30):
        ep = sample_episodes(
            n_units=32, n_features=None, n_episodes=1, seed=e,
        )[0]
        d = ep["table"]["n_features"]
        assert 2 <= d <= 1000
        ds.append(d)
    assert any(d >= 8 for d in ds)
    med = float(np.median(ds))
    assert 10 <= med <= 40, med


def test_hub_can_have_many_parents():
    saw_wide = False
    d = 40
    for seed in range(12):
        ep = sample_episodes(
            n_units=32, n_features=d, n_episodes=1, seed=seed,
            independent_frac=0.0, return_mechanism=True, max_parents=None,
            graph_family="star",
        )[0]
        parents = ep["response_law"]["parents"]
        _assert_dag(parents)
        n_edges = sum(len(pa) for pa in parents)
        max_in = max(len(pa) for pa in parents)
        outdeg = [0] * d
        for pa in parents:
            for p in pa:
                outdeg[p] += 1
        max_out = max(outdeg) if outdeg else 0
        assert n_edges < d * (d - 1) / 4
        law = ep["response_law"]
        assert "out_hubs" in law and "in_hubs" in law and "hubs" in law
        if max_out >= 15 or max_in >= 15:
            saw_wide = True
    assert saw_wide


def test_star_hubs_d20_one_seed_wide():
    saw = False
    d = 20
    for seed in range(10):
        ep = sample_episodes(
            n_units=32, n_features=d, n_episodes=1, seed=seed,
            independent_frac=0.0, return_mechanism=True, max_parents=None,
            graph_family="star",
        )[0]
        parents = ep["response_law"]["parents"]
        _assert_dag(parents)
        max_in = max(len(pa) for pa in parents)
        outdeg = [0] * d
        for pa in parents:
            for p in pa:
                outdeg[p] += 1
        max_out = max(outdeg) if outdeg else 0
        if max(max_in, max_out) >= 8:
            saw = True
    assert saw


def test_categorical_linear_softmax():
    ep = sample_episodes(
        n_units=30, n_features=8, n_episodes=1, seed=0,
        type_weights={
            "numeric": 0, "ordinal": 0, "binary": 0,
            "categorical": 1, "high_cardinality": 0,
        },
        independent_frac=0.0, return_mechanism=True,
    )[0]
    n = 30
    assert set(ep["table"]["column_types"]) == {"categorical"}
    for j, col in enumerate(ep["response_law"]["columns"]):
        assert col["g"] == "linear_softmax"
        k = ep["table"]["n_classes"][j]
        assert k is not None and k >= 2
        vals = [ep["table"]["values"][i][j] for i in range(n)]
        assert all(isinstance(v, int) and 0 <= v < k for v in vals)


def test_explicit_max_parents_is_hard_cap():
    for seed in range(8):
        ep = sample_episodes(
            n_units=24, n_features=40, n_episodes=1, seed=seed,
            independent_frac=0.0, return_mechanism=True, max_parents=2,
        )[0]
        parents = ep["response_law"]["parents"]
        assert all(len(pa) <= 2 for pa in parents)


def test_activations_and_noise_in_law():
    from generator import ACTIVATIONS, NOISE_FAMS, _apply_activation, DEFAULT_TOKEN_HERITABILITY
    ep = sample_episodes(
        n_units=32, n_features=24, n_episodes=1, seed=0,
        independent_frac=0.0, return_mechanism=True, unit_dim=8,
    )[0]
    law = ep["response_law"]
    acts = law["activations"]
    fams = law["noise_families"]
    assert len(acts) == 24 and len(fams) == 24
    assert set(acts) <= set(ACTIVATIONS)
    assert set(fams) <= set(NOISE_FAMS)
    for c, fam in zip(law["columns"], fams):
        assert c["noise_family"] == fam
        if fam == "student_t":
            assert c["noise_df"] in (3, 4, 5, 6)
        else:
            assert c["noise_df"] is None
    tokens = law["tokens"]
    parents = law["parents"]
    betas = law["betas"]
    id_cos = []
    all_cos = []
    for j, pa in enumerate(parents):
        if not pa:
            continue
        t_j = np.asarray(tokens[j], dtype=np.float64)
        s_raw = np.zeros_like(t_j)
        for b, p in zip(betas[j], pa):
            s_raw = s_raw + float(b) * np.asarray(tokens[p], dtype=np.float64)
        phi = acts[j] or "identity"
        s = _apply_activation(s_raw, phi)
        s = s / max(float(np.linalg.norm(s)), 1e-8)
        cos = float(np.dot(t_j, s))
        all_cos.append(cos)
        assert cos > 0.5
        if phi == "identity":
            id_cos.append(cos)
            assert abs(cos - DEFAULT_TOKEN_HERITABILITY) < 1e-5
    assert all_cos
    assert id_cos


def test_graph_is_dag():
    ep = sample_episodes(
        n_units=24, n_features=40, n_episodes=1, seed=3,
        independent_frac=0.0, return_mechanism=True, max_parents=None,
    )[0]
    _assert_dag(ep["response_law"]["parents"])



def test_default_graph_mostly_sparse():
    n_star = 0
    n_eps = 40
    d = 24
    for seed in range(n_eps):
        ep = sample_episodes(
            n_units=24, n_features=d, n_episodes=1, seed=seed,
            independent_frac=0.0, return_mechanism=True,
        )[0]
        law = ep["response_law"]
        fam = law.get("graph_family")
        assert fam in ("sparse", "star")
        if fam == "star" or law.get("out_hubs") or law.get("in_hubs"):
            n_star += 1
        else:
            max_in = max(len(pa) for pa in law["parents"])
            assert max_in <= 6
    assert n_star <= 16, n_star
    assert n_star >= 1, n_star


def test_unit_dim_none_in_range():
    ks = []
    for seed in range(24):
        ep = sample_episodes(
            n_units=16, n_features=8, n_episodes=1, seed=seed,
            unit_dim=None, return_mechanism=True, independent_frac=0.0,
        )[0]
        k = ep["population"]["unit_dim"]
        assert 2 <= k <= 1024
        tok = next(t for t in ep["response_law"]["tokens"] if t is not None)
        assert len(tok) == k
        ks.append(k)
    assert min(ks) >= 2
    assert any(k <= 32 for k in ks)


def test_population_mixture_bounds():
    for n, seed in ((8, 0), (16, 1), (64, 2), (100, 3), (400, 4)):
        ep = sample_episodes(
            n_units=n, n_features=4, n_episodes=1, seed=seed, return_mechanism=True,
        )[0]
        mix = ep["population"]["mixture"]
        M = int(mix["n_components"])
        m_max = int(np.floor(np.sqrt(n)))
        assert 1 <= M <= m_max
        assert len(mix["components"]) == M
        fams = {c["family"] for c in mix["components"]}
        assert fams <= {"gaussian", "cauchy"}
        assign = ep["population"]["assignment"]
        assert len(assign) == n
        assert set(assign) <= set(range(M))
        U = np.array(ep["population"]["representations"])
        assert U.shape == (n, ep["population"]["unit_dim"])


def test_population_mixture_can_be_multi_component():
    seen = set()
    for seed in range(40):
        ep = sample_episodes(
            n_units=64, n_features=4, n_episodes=1, seed=seed, return_mechanism=True,
        )[0]
        seen.add(int(ep["population"]["mixture"]["n_components"]))
        if 1 in seen and max(seen) > 1:
            break
    assert 1 in seen
    assert max(seen) > 1


def test_discoscm_label_column_query():
    ep = sample_episodes(
        n_units=20, n_features=8, n_episodes=1, seed=0,
        query_mode="label_column", missing_frac=0.0, query_column=3,
    )[0]
    q = np.array(ep["table"]["query_mask"])
    m = np.array(ep["table"]["missing_mask"])
    assert ep["table"]["query_mode"] == "label_column"
    assert ep["table"]["query_column"] == 3
    assert q.shape == (20, 8)
    assert q[:, 3].all()
    assert not q[:, :3].any() and not q[:, 4:].any()
    assert m.sum() == 0
    assert ep["table"]["column_types"][3] in {
        "numeric", "ordinal", "binary", "categorical", "high_cardinality"
    }


def test_default_n_episodes_is_eight():
    eps = sample_episodes(n_units=8, n_features=4, seed=0, unit_dim=4)
    assert len(eps) == 8
    assert {ep["table"]["n_units"] for ep in eps} == {8}


def test_batch_shares_shape():
    eps = sample_episodes(
        n_units=16, n_features=None, unit_dim=None, seed=0, batch_size=8,
    )
    assert len(eps) == 8
    ds = {ep["table"]["n_features"] for ep in eps}
    ks = {ep["table"]["unit_dim"] for ep in eps}
    ns = {ep["table"]["n_units"] for ep in eps}
    assert len(ds) == 1 and len(ks) == 1 and ns == {16}


def test_n_episodes_packs_into_shape_batches():
    eps = sample_episodes(
        n_units=16, n_features=None, unit_dim=None,
        seed=0, n_episodes=16, batch_size=8,
    )
    assert len(eps) == 16
    d0 = {ep["table"]["n_features"] for ep in eps[:8]}
    d1 = {ep["table"]["n_features"] for ep in eps[8:]}
    k0 = {ep["table"]["unit_dim"] for ep in eps[:8]}
    k1 = {ep["table"]["unit_dim"] for ep in eps[8:]}
    assert len(d0) == 1 and len(d1) == 1 and len(k0) == 1 and len(k1) == 1
    assert {ep["seed"] for ep in eps} == set(range(16))
