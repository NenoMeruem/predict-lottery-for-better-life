import numpy as np
import pytest

from conftest import random_df
from src import predictor as p
from src.games import get_spec
from src.ml import rf_model
from src.ml.features import FEATURE_NAMES, MIN_HISTORY, build_features, training_set


def _P(game="535", n=150, seed=0):
    spec = get_spec(game)
    return p.main_presence(random_df(game, n_draws=n, seed=seed), spec), spec


def test_feature_shape_and_names():
    P, spec = _P()
    X = build_features(P, spec)
    assert X.shape == (len(P) + 1, spec.max_number, len(FEATURE_NAMES))
    assert np.isfinite(X).all()


@pytest.mark.parametrize("game", ["535", "645", "655"])
def test_features_have_no_lookahead(game):
    """X[t] tính từ toàn bộ dữ liệu phải bằng X[t] tính chỉ từ P[:t]."""
    P, spec = _P(game, n=120)
    X = build_features(P, spec)
    for t in (1, 10, 31, 60, 119, 120):
        assert np.allclose(X[t], build_features(P[:t], spec)[t]), t


def test_future_change_does_not_affect_past_features():
    P, spec = _P()
    P2 = P.copy()
    P2[100:] = P2[100:][::-1]  # đảo kết quả từ kỳ 100 trở đi
    X1, X2 = build_features(P, spec), build_features(P2, spec)
    assert np.allclose(X1[:101], X2[:101])


def test_known_feature_values():
    spec = get_spec("535")
    P = np.zeros((4, spec.max_number), dtype=bool)
    P[0, [0, 1, 2, 3, 4]] = P[1, [0, 1, 2, 3, 5]] = P[2, [0, 1, 2, 3, 6]] = True
    X = build_features(P, spec)
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    assert X[3, 0, i["current_gap"]] == 0  # số 1 về ở kỳ gần nhất
    assert X[3, 4, i["current_gap"]] == 2  # số 5 về ở kỳ 0, chưa về 2 kỳ
    assert X[3, 20, i["current_gap"]] == 3  # số 21 chưa từng về
    assert X[3, 0, i["freq_5"]] == pytest.approx(3 / 3)
    assert X[3, 4, i["overall_freq"]] == pytest.approx(1 / 3)
    assert X[2, 0, i["prev_sum"]] == 1 + 2 + 3 + 4 + 6


def test_training_set_uses_only_labels_before_end():
    P, spec = _P(n=100)
    X = build_features(P, spec)
    xs, ys = training_set(X, P, end=60)
    assert len(ys) == (60 - MIN_HISTORY) * spec.max_number
    assert ys.sum() == P[MIN_HISTORY:60].sum()


def test_fit_returns_none_without_enough_history():
    P, spec = _P(n=20)
    assert rf_model.fit(build_features(P, spec), P, end=len(P)) is None
    assert (rf_model.next_draw_weights(P, spec) == 1).all()


def test_weights_positive_and_ranked_by_probability():
    w = rf_model.to_weights(np.array([0.10, 0.14, 0.18]))
    assert (w > 0).all() and w[0] < w[1] < w[2]


def test_next_draw_weights_and_generate_rf():
    P, spec = _P(n=150)
    w = rf_model.next_draw_weights(P, spec, seed=0)
    assert w.shape == (spec.max_number,) and (w > 0).all()
    df = random_df("535", n_draws=150)
    t1 = p.generate(df, spec, "rf", 4, seed=3)
    assert t1 == p.generate(df, spec, "rf", 4, seed=3)
    assert len({t.numbers for t in t1}) == 4 and all(1 <= t.bonus <= 12 for t in t1)


def test_backtest_includes_rf_and_diagnostics():
    spec = get_spec("535")
    df = random_df("535", n_draws=160, seed=2)
    bt = p.backtest(df, spec, ["random", "rf"], n_draws=40, tickets_per_draw=3, retrain_every=20)
    assert bt["strategy"].tolist() == ["random", "rf"] and (bt["tickets"] == 120).all()
    d = rf_model.diagnostics(p.main_presence(df, spec), spec)
    assert 0.4 < d["auc"] < 0.6 and len(d["importances"]) == len(FEATURE_NAMES)
