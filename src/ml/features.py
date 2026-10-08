"""
Tạo đặc trưng cho mô hình Random Forest.

Mỗi mẫu là một cặp (kỳ t, số n). Nhãn y[t, n] = 1 nếu số n về ở kỳ t.
Đặc trưng của kỳ t CHỈ dùng dữ liệu của các kỳ < t (không rò rỉ dữ liệu tương lai):
    build_features(P)[t] chỉ phụ thuộc P[:t]  (được kiểm tra trong tests/test_rf.py)

P là ma trận bool D x N (xem predictor.main_presence).
"""
from __future__ import annotations

import numpy as np

from ..games import GameSpec

WINDOWS = (5, 10, 20, 50, 100)

FEATURE_NAMES: list[str] = (
    ["number", "is_even", "is_low", "current_gap", "avg_gap", "max_gap", "overall_freq"]
    + [f"freq_{w}" for w in WINDOWS]
    + ["trend_10_vs_50", "pair_affinity_prev", "prev_sum", "prev_odd_count"]
)

MIN_HISTORY = 30  # số kỳ tối thiểu trước khi một kỳ được dùng để huấn luyện


def build_features(P: np.ndarray, spec: GameSpec) -> np.ndarray:
    """Trả về mảng X có shape (D + 1, N, F).

    X[t] là đặc trưng để dự đoán kỳ t, tính từ P[:t]. X[D] dùng để dự đoán kỳ kế tiếp (chưa quay).
    """
    d, n = P.shape
    k = spec.n_main
    numbers = np.arange(1, n + 1)
    X = np.zeros((d + 1, n, len(FEATURE_NAMES)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    cs = np.zeros((d + 1, n), dtype=np.int32)  # cs[t] = số lần về trong P[:t]
    np.cumsum(P, axis=0, out=cs[1:])

    last_seen = np.full(n, -1)  # chỉ số kỳ gần nhất số này về (trước t)
    max_gap = np.zeros(n)
    co = np.zeros((n, n), dtype=np.float32)  # đồng xuất hiện trong P[:t]
    half = n // 2

    for t in range(d + 1):
        f = X[t]
        f[:, idx["number"]] = numbers
        f[:, idx["is_even"]] = numbers % 2 == 0
        f[:, idx["is_low"]] = numbers <= half

        gap = np.where(last_seen >= 0, t - 1 - last_seen, t)  # số kỳ liên tiếp chưa về
        cur_max = np.maximum(max_gap, gap)
        f[:, idx["current_gap"]] = gap
        f[:, idx["max_gap"]] = cur_max
        cnt = cs[t]
        f[:, idx["avg_gap"]] = t / np.maximum(cnt, 1)
        f[:, idx["overall_freq"]] = cnt / max(t, 1)
        for w in WINDOWS:
            f[:, idx[f"freq_{w}"]] = (cnt - cs[max(0, t - w)]) / max(min(w, t), 1)
        f[:, idx["trend_10_vs_50"]] = f[:, idx["freq_10"]] - f[:, idx["freq_50"]]

        if t > 0:
            prev = np.flatnonzero(P[t - 1])
            cond = co[:, prev] / np.maximum(cnt[prev], 1)  # P(n về | m đã về)
            cond[prev, np.arange(len(prev))] = 0.0  # bỏ chính nó
            f[:, idx["pair_affinity_prev"]] = cond.mean(axis=1) if len(prev) else 0.0
            f[:, idx["prev_sum"]] = (prev + 1).sum()
            f[:, idx["prev_odd_count"]] = ((prev + 1) % 2 == 1).sum()
        else:
            f[:, idx["pair_affinity_prev"]] = 0.0
            f[:, idx["prev_sum"]] = k * (n + 1) / 2
            f[:, idx["prev_odd_count"]] = k / 2

        # cập nhật trạng thái bằng kỳ t để dùng cho kỳ t + 1
        if t < d:
            row = P[t]
            max_gap = cur_max
            drawn = np.flatnonzero(row)
            last_seen[drawn] = t
            co[np.ix_(drawn, drawn)] += 1
    return X


def training_set(X: np.ndarray, P: np.ndarray, end: int, min_history: int = MIN_HISTORY) -> tuple[np.ndarray, np.ndarray]:
    """Mẫu huấn luyện từ các kỳ t trong [min_history, end), nhãn lấy từ P[t] (đã biết vì t < end)."""
    start = min(min_history, end)
    xs = X[start:end].reshape(-1, X.shape[2])
    ys = P[start:end].reshape(-1).astype(np.int8)
    return xs, ys
