import numpy as np
import pytest

from conftest import random_df
from src import predictor as p
from src.games import get_spec


@pytest.mark.parametrize("game", ["535", "645", "655"])
@pytest.mark.parametrize("strategy", list(p.STRATEGIES))
def test_generate_valid_tickets(game, strategy):
    spec = get_spec(game)
    tickets = p.generate(random_df(game), spec, strategy, n_tickets=6, seed=1)
    assert len(tickets) == 6
    assert len({t.numbers for t in tickets}) == 6  # không trùng vé
    for t in tickets:
        assert len(t.numbers) == spec.n_main == len(set(t.numbers))
        assert list(t.numbers) == sorted(t.numbers)
        assert all(1 <= n <= spec.max_number for n in t.numbers)
        if spec.bonus_player_picks:
            assert 1 <= t.bonus <= spec.bonus_range_max
        else:
            assert t.bonus is None


def test_generate_reproducible_with_seed():
    spec, df = get_spec("535"), random_df("535")
    assert p.generate(df, spec, "hot", 3, seed=7) == p.generate(df, spec, "hot", 3, seed=7)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        p.generate(random_df("645"), get_spec("645"), "magic")


def test_weights():
    # 4 kỳ x 3 số; số 1 về liên tục, số 3 chưa về lần nào
    h = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool)
    assert p.strategy_weights("hot", h).argmax() == 0
    assert p.strategy_weights("cold", h).argmax() == 2
    assert p.strategy_weights("overdue", h).tolist() == [1.0, 1.0, 5.0]
    assert (p.strategy_weights("random", h) == 1).all()


def test_expected_matches():
    mu, _ = p.expected_matches(get_spec("535"))
    assert mu == pytest.approx(5 * 5 / 35)


def test_backtest_structure_and_baseline():
    spec = get_spec("535")
    df = random_df("535", n_draws=300, seed=3)
    bt = p.backtest(df, spec, ["random", "hot", "pairs"], n_draws=100, tickets_per_draw=4, seed=0)
    assert bt["strategy"].tolist() == ["random", "hot", "pairs"]
    assert (bt["tickets"] == 400).all()
    hit_cols = [f"hit_{k}" for k in range(spec.n_main + 1)]
    assert (bt[hit_cols].sum(axis=1) == bt["tickets"]).all()
    assert "bonus_hits" in bt and bt.attrs["n_draws"] == 100
    # dữ liệu ngẫu nhiên: không chiến lược nào lệch quá xa mức ngẫu nhiên
    assert (bt["mean_matches"] - bt["expected"]).abs().max() < 0.2


def test_backtest_does_not_peek_at_future():
    """Nếu backtest vô tình dùng kết quả của chính kỳ đang thử, 'hot' với window=1 sẽ trùng rất nhiều."""
    spec = get_spec("645")
    df = random_df("645", n_draws=200, seed=5)
    bt = p.backtest(df, spec, ["hot"], n_draws=100, tickets_per_draw=3, window=1)
    assert bt["mean_matches"].iloc[0] < 1.5
