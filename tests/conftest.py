import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaner import clean  # noqa: E402
from src.games import GameSpec, get_spec  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def make_raw(draws, bonus=None, start="2020-01-01") -> pd.DataFrame:
    """Tạo DataFrame thô (dạng chuỗi như CSV) từ danh sách bộ số."""
    dates = pd.date_range(start, periods=len(draws), freq="2D")
    rows = []
    for i, (nums, d) in enumerate(zip(draws, dates)):
        row = {"draw_id": f"{i + 1:05d}", "date": d.strftime("%d/%m/%Y")}
        row.update({f"n{j + 1}": f"{n:02d}" for j, n in enumerate(nums)})
        if bonus is not None:
            row["bonus"] = f"{bonus[i]:02d}"
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_spec() -> GameSpec:
    return GameSpec("t36", "Test 3/6", n_main=3, max_number=6)


@pytest.fixture
def tiny_df(tiny_spec):
    df, _ = clean(make_raw([[1, 2, 3], [1, 2, 4], [1, 5, 6], [2, 3, 4]]), tiny_spec)
    return df


def random_df(game: str, n_draws: int = 120, seed: int = 0) -> pd.DataFrame:
    spec = get_spec(game)
    rng = random.Random(seed)
    draws, bonus = [], []
    for _ in range(n_draws):
        nums = rng.sample(list(spec.numbers), spec.n_main + 1)
        draws.append(nums[:-1])
        bonus.append(rng.randint(1, spec.bonus_range_max) if spec.bonus_separate else nums[-1])
    df, _ = clean(make_raw(draws, bonus if spec.has_bonus else None, start="2024-06-01"), spec)
    return df
