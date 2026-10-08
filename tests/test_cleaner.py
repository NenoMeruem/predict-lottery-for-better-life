import pandas as pd

from conftest import FIXTURES
from src.cleaner import clean, load_clean, load_processed, load_raw, save_processed, to_long
from src.games import get_spec


def test_clean_645_drops_invalid_rows():
    df, report = clean(load_raw("645", FIXTURES), get_spec("645"))
    assert report.total_rows == 10
    assert report.kept_rows == 4
    assert df["draw_id"].tolist() == [1, 2, 7, 8]
    reasons = dict(report.dropped)
    assert "thiếu số chính" in reasons["00003"]
    assert "ngoài khoảng" in reasons["00004"]
    assert "trùng trong cùng kỳ" in reasons["00005"]
    assert "ngày" in reasons["00006"]
    assert "trùng draw_id" in reasons["00007"]
    assert "draw_id" in reasons[""]


def test_clean_types_sorting_and_features():
    df, _ = clean(load_raw("645", FIXTURES), get_spec("645"))
    assert str(df["draw_id"].dtype) == "Int64"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    # kỳ 7: giữ bản sau (45,01,22,13,07,30), số được sắp xếp tăng dần
    row = df[df["draw_id"] == 7].iloc[0]
    assert [row[f"n{i}"] for i in range(1, 7)] == [1, 7, 13, 22, 30, 45]
    assert row["date"] == pd.Timestamp("2020-01-17")
    assert row["sum"] == 118
    assert row["odd_count"] == 4 and row["even_count"] == 2
    assert row["low_count"] == 4  # 1, 7, 13, 22 <= 22
    # khoảng trắng được loại bỏ
    assert df[df["draw_id"] == 8].iloc[0]["n1"] == 5


def test_clean_655_bonus_rules():
    df, report = clean(load_raw("655", FIXTURES), get_spec("655"))
    assert df["draw_id"].tolist() == [1, 2]
    assert pd.isna(df.loc[1, "bonus"])
    assert report.missing_bonus == 1
    reasons = dict(report.dropped)
    assert "bonus trùng" in reasons["00003"]
    assert "bonus ngoài khoảng" in reasons["00004"]


def test_to_long(tiny_df, tiny_spec):
    long = to_long(tiny_df, tiny_spec)
    assert len(long) == len(tiny_df) * tiny_spec.n_main
    assert set(long.columns) == {"draw_id", "date", "position", "number"}


def test_save_and_load_processed(tmp_path):
    raw_dir = tmp_path / "data"
    raw_dir.mkdir()
    (raw_dir / "645.csv").write_text((FIXTURES / "645.csv").read_text())
    df, _ = load_clean("645", raw_dir, save=True)
    assert (raw_dir / "processed" / "645.parquet").exists()
    assert (raw_dir / "processed" / "645.csv").exists()
    back = load_processed("645", raw_dir)
    pd.testing.assert_frame_equal(df, back, check_dtype=False)


def test_clean_535_separate_bonus_pool():
    df, report = clean(load_raw("535", FIXTURES), get_spec("535"))
    assert df["draw_id"].tolist() == [1, 4]
    assert df.loc[0, "bonus"] == 9  # bonus trùng số chính vẫn hợp lệ (lồng riêng)
    assert report.missing_bonus == 1
    reasons = dict(report.dropped)
    assert "bonus ngoài khoảng 1..12" in reasons["00002"]
    assert "ngoài khoảng 1..35" in reasons["00003"]
