import pytest

from conftest import random_df
from src.cli import main


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for game in ("535", "645"):
        df = random_df(game, n_draws=80)
        out = df[["draw_id", "date", *[c for c in df.columns if c.startswith("n") or c == "bonus"]]].copy()
        out["draw_id"] = out["draw_id"].map("{:05d}".format)
        out["date"] = out["date"].dt.strftime("%d/%m/%Y")
        out.to_csv(d / f"{game}.csv", index=False)
    return d


def test_predict(data_dir, capsys):
    assert main(["predict", "535", "-d", str(data_dir), "-s", "hot", "-s", "pairs", "-n", "3", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "[hot]" in out and "[pairs]" in out and "[random]" not in out
    assert out.count(" | ") >= 6  # vé 535 có số đặc biệt


def test_backtest_writes_report(data_dir, tmp_path, capsys):
    reports = tmp_path / "reports"
    assert main(["backtest", "535", "-d", str(data_dir), "-o", str(reports), "--draws", "20", "-n", "2"]) == 0
    assert (reports / "535" / "backtest.md").exists()
    assert (reports / "535" / "backtest.png").exists()
    assert "backtest.md" in (reports / "README.md").read_text(encoding="utf-8")


def test_legacy_report_syntax(data_dir, capsys):
    assert main(["645", "-d", str(data_dir), "--no-plots"]) == 0
    assert "Mega 6/45" in capsys.readouterr().out


def test_default_games_are_those_with_data(data_dir, capsys):
    assert main(["predict", "-d", str(data_dir), "-s", "random", "-n", "1"]) == 0
    out = capsys.readouterr().out
    assert "Lotto 5/35" in out and "Mega 6/45" in out and "Power 6/55" not in out


def test_missing_data(data_dir, capsys):
    assert main(["predict", "655", "-d", str(data_dir)]) == 2
    assert "Chưa có dữ liệu" in capsys.readouterr().err
