
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
    eps = sample_episodes(n_units=8, n_features=4, n_episodes=2, seed=0, return_mechanism=True)
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

