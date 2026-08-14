import numpy as np
from generator import sample_episode, sample_episodes

def test_masks_partition_grid():
    rng = np.random.default_rng(0)
    ep = sample_episode(rng, n_units=16, n_features=4, seed=0)
    q = np.array(ep["query_mask"])
    c = np.array(ep["context_mask"])
    assert q.shape == (16, 4)
    assert not np.any(q & c)
    assert np.all(q | c)
    assert int(q.sum()) == len(ep["y_query"])
    assert ep["y_input"][0].count(None) + sum(row.count(None) for row in ep["y_input"][1:]) >= 1

def test_mechanism_omitted_by_default():
    eps = sample_episodes(n_units=8, n_features=4, n_episodes=1, seed=1)
    ep = eps[0]
    assert "mechanism" not in ep
    assert "U" not in ep and "W" not in ep and "Y_full" not in ep

def test_return_mechanism_is_discoscm_tuple():
    eps = sample_episodes(
        n_units=8, n_features=4, unit_dim=3, n_episodes=1, seed=1, return_mechanism=True
    )
    ep = eps[0]
    m = ep["mechanism"]
    assert m["framework"] == "DiscoSCM"
    assert m["observational"] is True
    assert m["V"] == ["Y"]
    U = np.array(m["U"]["values"])
    W = np.array(m["F"]["W"]["values"])
    E = np.array(m["E"]["values"])
    assert U.shape == (8, 3)
    assert W.shape == (4, 3)
    assert E.shape == (8, 4)
    Y = U @ W.T + E
    # query values must match the law
    q = np.array(ep["query_mask"])
    yq = np.array(ep["y_query"])
    assert np.allclose(Y[q], yq)
    assert "Y_full" not in ep  # mechanism flag alone does not dump the world table

def test_debug_implies_mechanism_and_world():
    eps = sample_episodes(n_units=8, n_features=4, n_episodes=1, seed=1, debug=True)
    ep = eps[0]
    assert "mechanism" in ep
    assert np.array(ep["Y_full"]).shape == (8, 4)
