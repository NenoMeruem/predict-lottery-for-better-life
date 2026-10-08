import math

import numpy as np
import pytest

from conftest import make_raw, random_df
from src import analyzer as a
from src.cleaner import clean
from src.games import get_spec

# tiny_df (Test 3/6): [1,2,3], [1,2,4], [1,5,6], [2,3,4]


def test_frequency(tiny_df, tiny_spec):
    f = a.frequency(tiny_df, tiny_spec).set_index("number")
    assert f["count"].to_dict() == {1: 3, 2: 3, 3: 2, 4: 2, 5: 1, 6: 1}
    assert f["count"].sum() == len(tiny_df) * tiny_spec.n_main
    assert f["expected"].iloc[0] == pytest.approx(4 * 3 / 6)
    assert f.loc[1, "pct"] == pytest.approx(75.0)


def test_frequency_includes_never_drawn_numbers():
    df = random_df("645", n_draws=3)
    f = a.frequency(df, get_spec("645"))
    assert len(f) == 45 and (f["count"] == 0).any()


def test_bonus_frequency():
    spec = get_spec("655")
    df = random_df("655", n_draws=50)
    f = a.frequency(df, spec, source="bonus")
    assert f["count"].sum() == 50
    with pytest.raises(ValueError):
        a.frequency(random_df("645", 5), get_spec("645"), source="bonus")


def test_pairs(tiny_df, tiny_spec):
    pm = a.pair_matrix(tiny_df, tiny_spec)
    assert (pm.to_numpy() == pm.to_numpy().T).all()
    assert pm.loc[1, 2] == 2 and pm.loc[2, 4] == 2 and pm.loc[3, 5] == 0
    assert np.diag(pm).tolist() == [3, 3, 2, 2, 1, 1]
    pairs = a.pair_frequency(tiny_df, tiny_spec)
    assert len(pairs) == math.comb(6, 2)
    assert pairs["count"].sum() == len(tiny_df) * math.comb(3, 2)
    assert pairs.iloc[0]["count"] == 2


def test_triplets(tiny_df, tiny_spec):
    t = a.triplet_frequency(tiny_df, tiny_spec, top=10)
    assert len(t) == 4 and (t["count"] == 1).all()


def test_gaps(tiny_df, tiny_spec):
    g = a.gaps(tiny_df, tiny_spec).set_index("number")
    assert g.loc[1, "current_gap"] == 1 and g.loc[1, "avg_gap"] == 0
    assert g.loc[3, "current_gap"] == 0 and g.loc[3, "avg_gap"] == 2 and g.loc[3, "max_gap"] == 2
    assert g.loc[5, "max_gap"] == 2 and np.isnan(g.loc[5, "avg_gap"])
    assert g.loc[1, "last_seen_draw"] == 3


def test_hot_cold(tiny_df, tiny_spec):
    hc = a.hot_cold(tiny_df, tiny_spec, window=2, top=1)
    assert hc.attrs["window"] == 2
    assert hc.loc[hc["status"] == "hot", "number"].item() in (1, 2)
    assert hc.loc[hc["status"] == "cold", "number"].item() in (5, 6)


def test_frequency_by_period():
    spec = get_spec("645")
    df = random_df("645", n_draws=300)  # 2024 -> 2026
    by_year = a.frequency_by_period(df, spec, "year", normalize=False)
    assert by_year.shape[0] == 45
    assert by_year.to_numpy().sum() == 300 * 6


def test_composition_expected_sums_to_100():
    spec = get_spec("655")
    d = a.composition_distribution(random_df("655"), spec, "odd")
    assert d["expected_pct"].sum() == pytest.approx(100)
    assert d["count"].sum() == 120


def test_uniformity_perfect(tiny_spec):
    df, _ = clean(make_raw([[1, 2, 3], [4, 5, 6]] * 10), tiny_spec)
    res = a.uniformity_test(df, tiny_spec)
    assert res["chi2"] == pytest.approx(0) and res["p_value"] == pytest.approx(1) and res["is_uniform"]


def test_uniformity_biased(tiny_spec):
    df, _ = clean(make_raw([[1, 2, 3]] * 30 + [[4, 5, 6]]), tiny_spec)
    assert not a.uniformity_test(df, tiny_spec)["is_uniform"]


def test_summary_runs_on_real_specs():
    for g in ("645", "655"):
        s = a.summary(random_df(g), get_spec(g))
        assert s["draws"] == 120 and len(s["hot"]) == 10
