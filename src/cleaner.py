"""
Tiền xử lý dữ liệu kết quả xổ số: chuẩn hoá định dạng, xử lý giá trị thiếu,
loại dòng không hợp lệ và ghi dữ liệu sạch ra data/processed/.

    from src.cleaner import load_clean
    df, report = load_clean("645")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .games import GameSpec, get_spec

DATE_FORMAT = "%d/%m/%Y"


@dataclass
class CleanReport:
    game: str
    total_rows: int = 0
    kept_rows: int = 0
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (draw_id gốc, lý do)
    missing_bonus: int = 0

    @property
    def dropped_rows(self) -> int:
        return len(self.dropped)

    def __str__(self) -> str:
        lines = [f"[{self.game}] {self.total_rows} dòng thô -> giữ {self.kept_rows}, loại {self.dropped_rows}"]
        if self.missing_bonus:
            lines.append(f"  - {self.missing_bonus} kỳ thiếu số bonus (giữ lại, để trống)")
        for draw_id, reason in self.dropped[:20]:
            lines.append(f"  - loại kỳ {draw_id or '?'}: {reason}")
        if self.dropped_rows > 20:
            lines.append(f"  ... và {self.dropped_rows - 20} dòng khác")
        return "\n".join(lines)


# ---------------------------------------------------------------- load

def raw_path(game: str, data_dir: str | Path = "data") -> Path:
    return Path(data_dir) / f"{game}.csv"


def load_raw(game: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Đọc CSV thô, giữ mọi cột ở dạng chuỗi để tự kiểm soát việc chuyển kiểu."""
    return pd.read_csv(raw_path(game, data_dir), dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- clean

def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.strip(), errors="coerce").astype("Int64")


def clean(raw: pd.DataFrame, spec: GameSpec) -> tuple[pd.DataFrame, CleanReport]:
    """Trả về (DataFrame sạch, CleanReport).

    Quy tắc:
    - Thiếu draw_id / date / bất kỳ số chính nào -> loại (không tự điền để tránh sai thống kê).
    - Số chính ngoài [1, max_number] hoặc trùng nhau trong cùng kỳ -> loại.
    - Bonus thiếu -> giữ, để <NA>. Bonus ngoài khoảng hoặc trùng số chính -> loại.
    - Trùng draw_id -> giữ bản cuối cùng.
    """
    report = CleanReport(game=spec.code, total_rows=len(raw))
    required = ["draw_id", "date", *spec.main_cols] + (["bonus"] if spec.has_bonus else [])
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"[{spec.code}] thiếu cột: {missing_cols}")

    raw = raw.reset_index(drop=True)
    orig_id = raw["draw_id"].astype(str).str.strip()
    df = pd.DataFrame(index=raw.index)
    df["draw_id"] = _to_int(raw["draw_id"])
    df["date"] = pd.to_datetime(raw["date"].str.strip(), format=DATE_FORMAT, errors="coerce")
    for c in spec.main_cols:
        df[c] = _to_int(raw[c])
    if spec.has_bonus:
        df["bonus"] = _to_int(raw["bonus"])

    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask: pd.Series, why: str) -> None:
        mask = mask.fillna(False).astype(bool) & (reason == "")
        reason[mask] = why

    main = df[spec.main_cols]
    flag(df["draw_id"].isna(), "thiếu/sai draw_id")
    flag(df["date"].isna(), "thiếu/sai ngày")
    flag(main.isna().any(axis=1), "thiếu số chính")
    flag(((main < 1) | (main > spec.max_number)).any(axis=1), f"số chính ngoài khoảng 1..{spec.max_number}")
    flag(main.nunique(axis=1) < spec.n_main, "số chính bị trùng trong cùng kỳ")
    if spec.has_bonus:
        b = df["bonus"]
        flag((b < 1) | (b > spec.bonus_range_max), f"bonus ngoài khoảng 1..{spec.bonus_range_max}")
        if not spec.bonus_separate:
            flag(main.eq(b, axis=0).any(axis=1) & b.notna(), "bonus trùng số chính")

    dup = df["draw_id"].duplicated(keep="last") & df["draw_id"].notna()
    flag(dup, "trùng draw_id (giữ bản sau)")

    bad = reason != ""
    report.dropped = list(zip(orig_id[bad], reason[bad]))
    df = df[~bad].copy()

    # Sắp xếp số chính tăng dần trong mỗi kỳ
    df[spec.main_cols] = np.sort(df[spec.main_cols].to_numpy(dtype=int), axis=1)
    df[spec.main_cols] = df[spec.main_cols].astype("Int64")
    df = df.sort_values("draw_id").reset_index(drop=True)

    if spec.has_bonus:
        report.missing_bonus = int(df["bonus"].isna().sum())

    df = add_features(df, spec)
    report.kept_rows = len(df)
    return df, report


def add_features(df: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    nums = df[spec.main_cols].to_numpy(dtype=int)
    half = spec.max_number // 2
    df = df.copy()
    df["year"] = df["date"].dt.year.astype("Int64")
    df["month"] = df["date"].dt.month.astype("Int64")
    df["weekday"] = df["date"].dt.dayofweek.astype("Int64")  # 0 = thứ Hai
    df["sum"] = nums.sum(axis=1)
    df["odd_count"] = (nums % 2 == 1).sum(axis=1)
    df["even_count"] = spec.n_main - df["odd_count"]
    df["low_count"] = (nums <= half).sum(axis=1)
    df["high_count"] = spec.n_main - df["low_count"]
    return df


def to_long(df: pd.DataFrame, spec: GameSpec, include_bonus: bool = False) -> pd.DataFrame:
    """Chuyển sang dạng dài: mỗi dòng là một số trúng (draw_id, date, position, number)."""
    cols = list(spec.main_cols) + (["bonus"] if include_bonus and spec.has_bonus else [])
    long = df.melt(id_vars=["draw_id", "date"], value_vars=cols, var_name="position", value_name="number")
    return long.dropna(subset=["number"]).sort_values(["draw_id", "position"]).reset_index(drop=True)


# ---------------------------------------------------------------- save / load processed

def processed_dir(data_dir: str | Path = "data") -> Path:
    return Path(data_dir) / "processed"


def save_processed(df: pd.DataFrame, game: str, data_dir: str | Path = "data") -> list[Path]:
    """Ghi ra <data_dir>/processed/<game>.parquet (giữ kiểu dữ liệu) và .csv (dễ đọc)."""
    out = processed_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq, csv = out / f"{game}.parquet", out / f"{game}.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False, date_format="%Y-%m-%d")
    return [pq, csv]


def load_processed(game: str, data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_parquet(processed_dir(data_dir) / f"{game}.parquet")


def load_clean(game: str, data_dir: str | Path = "data", save: bool = False) -> tuple[pd.DataFrame, CleanReport]:
    spec = get_spec(game)
    df, report = clean(load_raw(game, data_dir), spec)
    if save:
        save_processed(df, game, data_dir)
    return df, report
