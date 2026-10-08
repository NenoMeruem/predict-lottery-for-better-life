"""
Tính toán thống kê trên DataFrame đã làm sạch (từ cleaner.py).

Mọi hàm đều là hàm thuần: nhận (df, spec), trả về DataFrame/dict, không đọc file, không vẽ.

Lưu ý: các kỳ quay là ngẫu nhiên và độc lập. Số "nóng/lạnh" hay "lâu chưa về" chỉ mô tả
quá khứ, không làm thay đổi xác suất của kỳ tiếp theo.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .games import GameSpec


# ---------------------------------------------------------------- helpers

def _main_matrix(df: pd.DataFrame, spec: GameSpec) -> np.ndarray:
    return df[spec.main_cols].to_numpy(dtype=int)


def presence_matrix(df: pd.DataFrame, spec: GameSpec) -> np.ndarray:
    """Ma trận bool D x N: presence[i, n-1] = True nếu số n về ở kỳ thứ i (theo thứ tự df)."""
    nums = _main_matrix(df, spec)
    mat = np.zeros((len(df), spec.max_number), dtype=bool)
    mat[np.arange(len(df))[:, None], nums - 1] = True
    return mat


# ---------------------------------------------------------------- frequency

def frequency(df: pd.DataFrame, spec: GameSpec, source: str = "main") -> pd.DataFrame:
    """Tần suất xuất hiện của từng số 1..N (kể cả số chưa về lần nào).

    source="main": bộ số chính; source="bonus": số đặc biệt (chỉ 6/55).
    Cột: number, count, pct (% số kỳ có số này), expected, deviation (count - expected), ratio.
    """
    draws = len(df)
    if source == "main":
        values = _main_matrix(df, spec).ravel()
        p = spec.p_number
        numbers = spec.numbers
    elif source == "bonus":
        if not spec.has_bonus:
            raise ValueError(f"{spec.name} không có số bonus")
        values = df["bonus"].dropna().to_numpy(dtype=int)
        draws = len(values)
        p = 1 / spec.bonus_range_max
        numbers = spec.bonus_numbers
    else:
        raise ValueError("source phải là 'main' hoặc 'bonus'")

    counts = np.bincount(values, minlength=len(numbers) + 1)[1:]
    expected = draws * p
    out = pd.DataFrame({"number": list(numbers), "count": counts})
    out["pct"] = out["count"] / draws * 100 if draws else 0.0
    out["expected"] = expected
    out["deviation"] = out["count"] - expected
    out["ratio"] = out["count"] / expected if expected else np.nan
    return out


def hot_cold(df: pd.DataFrame, spec: GameSpec, window: int = 50, top: int = 10) -> pd.DataFrame:
    """So sánh tần suất trong `window` kỳ gần nhất với toàn bộ lịch sử.

    Cột: number, recent_count, recent_expected, overall_count, overall_pct, status ("hot"/"cold"/"").
    "hot" = `top` số về nhiều nhất trong window, "cold" = `top` số về ít nhất.
    """
    window = min(window, len(df))
    recent = frequency(df.tail(window), spec)
    overall = frequency(df, spec)
    out = pd.DataFrame({
        "number": overall["number"],
        "recent_count": recent["count"],
        "recent_expected": window * spec.p_number,
        "overall_count": overall["count"],
        "overall_pct": overall["pct"],
    })
    out["status"] = ""
    hot_idx = out.sort_values(["recent_count", "overall_count"], ascending=[False, False]).index[:top]
    cold_idx = out.sort_values(["recent_count", "overall_count"], ascending=[True, True]).index[:top]
    out.loc[hot_idx, "status"] = "hot"
    out.loc[cold_idx, "status"] = "cold"
    out.attrs["window"] = window
    return out


def gaps(df: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    """Khoảng cách giữa các lần xuất hiện của từng số.

    gap = số kỳ liên tiếp KHÔNG về giữa hai lần về.
    Cột: number, last_seen_draw, last_seen_date, current_gap (số kỳ gần nhất chưa về; 0 = vừa về ở kỳ mới nhất),
    avg_gap, max_gap, expected_gap (kỳ vọng của phân phối hình học = (1-p)/p).
    """
    mat = presence_matrix(df, spec)
    d = len(df)
    draw_ids = df["draw_id"].to_numpy()
    dates = df["date"].to_numpy()
    rows = []
    for n in spec.numbers:
        pos = np.flatnonzero(mat[:, n - 1])
        if len(pos) == 0:
            rows.append((n, pd.NA, pd.NaT, d, np.nan, d))
            continue
        between = np.diff(pos) - 1
        current = d - 1 - pos[-1]
        max_gap = max(int(between.max()) if len(between) else 0, int(current), int(pos[0]))
        rows.append((n, draw_ids[pos[-1]], dates[pos[-1]], int(current),
                     float(between.mean()) if len(between) else np.nan, max_gap))
    out = pd.DataFrame(rows, columns=["number", "last_seen_draw", "last_seen_date",
                                      "current_gap", "avg_gap", "max_gap"])
    p = spec.p_number
    out["expected_gap"] = (1 - p) / p
    return out


# ---------------------------------------------------------------- pairs / triplets

def pair_matrix(df: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    """Ma trận N x N đối xứng: số kỳ mà cặp (i, j) cùng về. Đường chéo = tần suất đơn."""
    mat = presence_matrix(df, spec).astype(np.int64)
    co = mat.T @ mat
    idx = pd.Index(list(spec.numbers), name="number")
    return pd.DataFrame(co, index=idx, columns=idx)


def pair_frequency(df: pd.DataFrame, spec: GameSpec, top: int | None = None) -> pd.DataFrame:
    """Danh sách cặp số hay về cùng nhau. Cột: a, b, count, expected, ratio."""
    co = pair_matrix(df, spec).to_numpy()
    i, j = np.triu_indices(spec.max_number, k=1)
    out = pd.DataFrame({"a": i + 1, "b": j + 1, "count": co[i, j]})
    out["expected"] = len(df) * spec.p_pair
    out["ratio"] = out["count"] / out["expected"]
    out = out.sort_values(["count", "a", "b"], ascending=[False, True, True]).reset_index(drop=True)
    return out.head(top) if top else out


def triplet_frequency(df: pd.DataFrame, spec: GameSpec, top: int = 20) -> pd.DataFrame:
    """Top bộ 3 số hay về cùng nhau. Cột: a, b, c, count."""
    counter: Counter = Counter()
    for row in _main_matrix(df, spec):
        counter.update(combinations(row, 3))
    rows = [(*k, v) for k, v in counter.most_common(top)]
    return pd.DataFrame(rows, columns=["a", "b", "c", "count"])


# ---------------------------------------------------------------- over time / distributions

def frequency_by_period(df: pd.DataFrame, spec: GameSpec, period: str = "year",
                        normalize: bool = True) -> pd.DataFrame:
    """Ma trận số (hàng) x kỳ thời gian (cột).

    normalize=True: chia cho số kỳ quay trong giai đoạn đó, tức tỉ lệ % kỳ có số này
    (vì số kỳ mỗi năm khác nhau).
    """
    if period not in df.columns:
        raise ValueError(f"Không có cột {period!r}")
    mat = presence_matrix(df, spec)
    groups = df[period].to_numpy()
    keys = sorted(pd.unique(groups))
    data = {}
    for k in keys:
        sel = groups == k
        counts = mat[sel].sum(axis=0)
        data[k] = counts / sel.sum() * 100 if normalize else counts
    out = pd.DataFrame(data, index=pd.Index(list(spec.numbers), name="number"))
    out.columns.name = period
    return out


def composition_distribution(df: pd.DataFrame, spec: GameSpec, kind: str = "odd") -> pd.DataFrame:
    """Phân phối số lượng số lẻ (kind="odd") hoặc số nhỏ <= N/2 (kind="low") trong mỗi kỳ.

    So sánh với kỳ vọng lý thuyết (phân phối siêu bội).
    Cột: k, count, pct, expected_pct.
    """
    if kind == "odd":
        col, good = "odd_count", (spec.max_number + 1) // 2
    elif kind == "low":
        col, good = "low_count", spec.max_number // 2
    else:
        raise ValueError("kind phải là 'odd' hoặc 'low'")
    ks = np.arange(spec.n_main + 1)
    counts = df[col].value_counts().reindex(ks, fill_value=0).to_numpy()
    expected = stats.hypergeom(spec.max_number, good, spec.n_main).pmf(ks) * 100
    return pd.DataFrame({"k": ks, "count": counts, "pct": counts / len(df) * 100, "expected_pct": expected})


def sum_stats(df: pd.DataFrame, spec: GameSpec) -> dict:
    """Thống kê tổng các số chính so với lý thuyết (rút không hoàn lại)."""
    n, k = spec.max_number, spec.n_main
    exp_mean = k * (n + 1) / 2
    exp_std = np.sqrt(k * (n - k) * (n + 1) / 12)
    s = df["sum"]
    return {"mean": float(s.mean()), "std": float(s.std()), "min": int(s.min()), "max": int(s.max()),
            "expected_mean": exp_mean, "expected_std": float(exp_std)}


# ---------------------------------------------------------------- tests

def uniformity_test(df: pd.DataFrame, spec: GameSpec, alpha: float = 0.05) -> dict:
    """Kiểm định chi-square: tần suất các số có lệch khỏi phân phối đều hay không?

    Vì mỗi kỳ rút k số không hoàn lại nên thống kê Pearson được hiệu chỉnh với hệ số
    (N-1)/(N-k) để xấp xỉ chi2 với N-1 bậc tự do.
    """
    freq = frequency(df, spec)
    n, k = spec.max_number, spec.n_main
    raw_chi2 = float((((freq["count"] - freq["expected"]) ** 2) / freq["expected"]).sum())
    chi2 = raw_chi2 * (n - 1) / (n - k)
    dof = n - 1
    p_value = float(stats.chi2.sf(chi2, dof))
    uniform = p_value >= alpha
    conclusion = (
        f"p ≥ {alpha}: không đủ bằng chứng cho thấy có số nào được ưu tiên, "
        "tần suất phù hợp với quay ngẫu nhiên đều."
        if uniform else
        f"p < {alpha}: tần suất lệch khỏi phân phối đều có ý nghĩa thống kê."
    )
    return {"chi2": chi2, "dof": dof, "p_value": p_value, "alpha": alpha,
            "is_uniform": uniform, "conclusion": conclusion}


# ---------------------------------------------------------------- summary

def summary(df: pd.DataFrame, spec: GameSpec, window: int = 50, top: int = 10) -> dict:
    freq = frequency(df, spec)
    hc = hot_cold(df, spec, window=window, top=top)
    gp = gaps(df, spec)
    last = df.iloc[-1]
    return {
        "game": spec.name,
        "draws": len(df),
        "first_draw": (int(df["draw_id"].iloc[0]), df["date"].iloc[0].date()),
        "last_draw": (int(last["draw_id"]), last["date"].date()),
        "last_numbers": [int(last[c]) for c in spec.main_cols],
        "last_bonus": (int(last["bonus"]) if spec.has_bonus and pd.notna(last["bonus"]) else None),
        "most_frequent": freq.nlargest(top, "count")["number"].tolist(),
        "least_frequent": freq.nsmallest(top, "count")["number"].tolist(),
        "hot": hc[hc["status"] == "hot"].sort_values("recent_count", ascending=False)["number"].tolist(),
        "cold": hc[hc["status"] == "cold"].sort_values("recent_count")["number"].tolist(),
        "longest_absent": gp.nlargest(top, "current_gap")[["number", "current_gap"]].values.tolist(),
        "top_pairs": [tuple(r) for r in pair_frequency(df, spec, top=top)[["a", "b", "count"]].values.tolist()],
        "uniformity": uniformity_test(df, spec),
        "window": hc.attrs["window"],
    }
