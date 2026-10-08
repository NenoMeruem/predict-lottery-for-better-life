import re

import pytest

from conftest import random_df
from src.games import get_spec
from src.visualizer import build_report


@pytest.mark.parametrize("game", ["535", "645", "655"])
def test_build_report_creates_markdown_with_existing_images(tmp_path, game):
    spec = get_spec(game)
    md_path = build_report(random_df(game, n_draws=150), spec, out_dir=tmp_path)
    assert md_path == tmp_path / game / "README.md"
    md = md_path.read_text(encoding="utf-8")

    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    assert len(images) >= 8
    for img in images:
        assert (md_path.parent / img).stat().st_size > 1000, img
    for html in ("frequency.html", "pair_heatmap.html"):
        assert html in md and (md_path.parent / html).exists()
    assert ("bonus_frequency.png" in md) == spec.has_bonus
