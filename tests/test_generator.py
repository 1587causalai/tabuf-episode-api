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

def test_debug_omitted_by_default():
    eps = sample_episodes(n_units=8, n_features=4, n_episodes=1, seed=1, debug=False)
    assert "U" not in eps[0] and "W" not in eps[0]

def test_debug_includes_latents():
    eps = sample_episodes(n_units=8, n_features=4, n_episodes=1, seed=1, debug=True)
    assert np.array(eps[0]["U"]).shape == (8, 4)
    assert np.array(eps[0]["W"]).shape == (4, 4)
