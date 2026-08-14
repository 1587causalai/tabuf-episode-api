
import numpy as np
from generator import sample_episode, sample_episodes


def _masks(ep):
    t = ep["table"]
    return (
        np.array(t["missing_mask"]),
        np.array(t["context_mask"]),
        np.array(t["query_mask"]),
    )


def test_three_masks_partition_grid():
    rng = np.random.default_rng(0)
    ep = sample_episode(rng, n_units=16, n_features=4, missing_frac=0.1, query_frac=0.15, seed=0)
    m, c, q = _masks(ep)
    assert m.shape == c.shape == q.shape == (16, 4)
    assert not np.any(m & c)
    assert not np.any(m & q)
    assert not np.any(c & q)
    assert np.all(m | c | q)
    assert int(q.sum()) == len(ep["table"]["y_query"])
    vals = ep["table"]["values"]
    for i in range(16):
        for j in range(4):
            if c[i, j]:
                assert vals[i][j] is not None
            else:
                assert vals[i][j] is None


def test_default_omits_population_and_law():
    ep = sample_episodes(n_units=8, n_features=4, n_episodes=2, seed=1)[0]
    assert "population" not in ep and "response_law" not in ep
    assert "missing_mask" in ep["table"]


def test_n_episodes_is_the_return_unit():
    eps = sample_episodes(n_units=8, n_features=4, n_episodes=3, seed=0)
    assert len(eps) == 3
    assert [e["seed"] for e in eps] == [0, 1, 2]


def test_return_mechanism_has_units_and_law():
    ep = sample_episodes(
        n_units=8, n_features=4, unit_dim=3, n_episodes=1, seed=2, return_mechanism=True
    )[0]
    pop, law = ep["population"], ep["response_law"]
    U = np.array(pop["representations"])
    W = np.array(law["W"]["values"])
    E = np.array(law["noise"]["values"])
    assert U.shape == (8, 3)
    Y = U @ W.T + E
    q = np.array(ep["table"]["query_mask"])
    assert np.allclose(Y[q], ep["table"]["y_query"])
    assert "Y_full" not in ep


def test_zero_missing_still_emits_mask():
    ep = sample_episodes(n_units=8, n_features=4, missing_frac=0.0, n_episodes=1, seed=0)[0]
    m = np.array(ep["table"]["missing_mask"])
    assert m.sum() == 0
