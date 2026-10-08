"""
Sinh bộ số gợi ý theo các chiến lược thống kê và backtest chúng trên lịch sử.

    from src.predictor import generate, backtest
    tickets = generate(df, spec, "hot", n_tickets=5, seed=1)
    result = backtest(df, spec, strategies=["random", "hot"], n_draws=200)

> [!WARNING]
> Kết quả xổ số là ngẫu nhiên và độc lập. Không chiến lược nào làm tăng xác suất trúng.
> Hàm backtest() có mục đích kiểm chứng điều đó: mọi chiến lược đều cho số trùng trung bình
> xấp xỉ mức chọn ngẫu nhiên (k*k/N).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .games import GameSpec

STRATEGIES: dict[str, str] = {
    "random": "Ngẫu nhiên đều (mốc so sánh)",
    "frequency": "Trọng số theo tần suất toàn bộ lịch sử",
    "hot": "Ưu tiên số về nhiều trong N kỳ gần nhất",
    "cold": "Ưu tiên số về ít trong N kỳ gần nhất",
    "overdue": "Ưu tiên số lâu chưa về",
    "pairs": "Chọn dần theo cặp số hay về cùng nhau",
    "balanced": "Ngẫu nhiên, lọc tổng và chẵn/lẻ ở mức điển hình",
    "rf": "Random Forest (đặc trưng: độ trễ, tần suất trượt, cặp số, kỳ trước)",
}


@dataclass(frozen=True)
class Ticket:
    numbers: tuple[int, ...]
    bonus: int | None = None

    def __str__(self) -> str:
        s = " ".join(f"{n:02d}" for n in self.numbers)
        return s + (f" | {self.bonus:02d}" if self.bonus is not None else "")


# ---------------------------------------------------------------- presence matrices

def main_presence(df: pd.DataFrame, spec: GameSpec) -> np.ndarray:
    """Ma trận bool D x N: số n có về ở kỳ i hay không."""
    nums = df[spec.main_cols].to_numpy(dtype=int)
    mat = np.zeros((len(df), spec.max_number), dtype=bool)
    mat[np.arange(len(df))[:, None], nums - 1] = True
    return mat


def bonus_presence(df: pd.DataFrame, spec: GameSpec) -> np.ndarray:
    """Ma trận bool D x B cho số bonus (bỏ các kỳ thiếu bonus)."""
    b = df["bonus"].dropna().to_numpy(dtype=int)
    mat = np.zeros((len(b), spec.bonus_range_max), dtype=bool)
    mat[np.arange(len(b)), b - 1] = True
    return mat


# ---------------------------------------------------------------- strategies

def strategy_weights(strategy: str, history: np.ndarray, window: int = 50) -> np.ndarray:
    """Trọng số chọn cho từng số dựa trên ma trận lịch sử (D x K). Luôn > 0."""
    d, k = history.shape
    if strategy in ("random", "balanced") or d == 0:
        return np.ones(k)
    if strategy in ("frequency", "pairs"):
        return history.sum(axis=0) + 1.0
    recent = history[-window:].sum(axis=0)
    if strategy == "hot":
        return (recent + 1.0) ** 2
    if strategy == "cold":
        return 1.0 / (recent + 1.0) ** 2
    if strategy == "overdue":
        seen = history.any(axis=0)
        gap = np.where(seen, np.argmax(history[::-1], axis=0), d)  # số kỳ kể từ lần về gần nhất
        return gap + 1.0
    raise ValueError(f"Chiến lược không hỗ trợ: {strategy!r}. Chọn một trong {list(STRATEGIES)}")


def _weighted_sample(w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(w), size=k, replace=False, p=w / w.sum()) + 1


def _pick_pairs(history: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Số đầu chọn theo tần suất; các số sau chọn theo mức đồng xuất hiện với các số đã chọn."""
    h = history.astype(np.int64)
    co = h.T @ h
    freq = np.diag(co).astype(float) + 1.0
    picked = [int(_weighted_sample(freq, 1, rng)[0]) - 1]
    while len(picked) < k:
        w = co[picked].sum(axis=0).astype(float) + 1.0
        w[picked] = 0.0
        picked.append(int(rng.choice(len(w), p=w / w.sum())))
    return np.array(picked) + 1


def _pick_balanced(spec: GameSpec, rng: np.random.Generator, max_tries: int = 2000) -> np.ndarray:
    """Ngẫu nhiên đều, giữ lại bộ có tổng trong [TB ± 1 độ lệch chuẩn] và số lẻ ở mức phổ biến nhất."""
    n, k = spec.max_number, spec.n_main
    mean = k * (n + 1) / 2
    std = np.sqrt(k * (n - k) * (n + 1) / 12)
    odd_ok = {k // 2, (k + 1) // 2}
    nums = None
    for _ in range(max_tries):
        nums = rng.choice(n, size=k, replace=False) + 1
        if abs(nums.sum() - mean) <= std and int((nums % 2).sum()) in odd_ok:
            break
    return nums


def _pick_main(strategy: str, history: np.ndarray, spec: GameSpec, rng: np.random.Generator,
               window: int, weights: np.ndarray | None = None) -> tuple[int, ...]:
    if weights is not None:
        nums = _weighted_sample(weights, spec.n_main, rng)
    elif strategy == "pairs" and len(history):
        nums = _pick_pairs(history, spec.n_main, rng)
    elif strategy == "balanced":
        nums = _pick_balanced(spec, rng)
    else:
        nums = _weighted_sample(strategy_weights(strategy, history, window), spec.n_main, rng)
    return tuple(sorted(int(x) for x in nums))


def _pick_bonus(strategy: str, history: np.ndarray, rng: np.random.Generator, window: int) -> int:
    # pairs/balanced/rf không có ý nghĩa với 1 số bonus -> dùng tần suất / ngẫu nhiên
    s = {"pairs": "frequency", "balanced": "random", "rf": "frequency"}.get(strategy, strategy)
    return int(_weighted_sample(strategy_weights(s, history, window), 1, rng)[0])


def _generate_from(strategy: str, main_hist: np.ndarray, bonus_hist: np.ndarray | None, spec: GameSpec,
                   n_tickets: int, rng: np.random.Generator, window: int,
                   weights: np.ndarray | None = None) -> list[Ticket]:
    tickets: list[Ticket] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(n_tickets * 20):
        nums = _pick_main(strategy, main_hist, spec, rng, window, weights)
        if nums in seen:
            continue
        seen.add(nums)
        bonus = _pick_bonus(strategy, bonus_hist, rng, window) if bonus_hist is not None else None
        tickets.append(Ticket(nums, bonus))
        if len(tickets) == n_tickets:
            break
    return tickets


def generate(df: pd.DataFrame, spec: GameSpec, strategy: str = "random", n_tickets: int = 5,
             window: int = 50, seed: int | None = None) -> list[Ticket]:
    """Sinh `n_tickets` bộ số (không trùng nhau) cho kỳ tiếp theo theo `strategy`.

    Với game mà người chơi tự chọn bonus (Lotto 5/35), mỗi vé có thêm số bonus.
    Chiến lược "rf" huấn luyện Random Forest trên toàn bộ lịch sử rồi lấy mẫu theo xác suất dự đoán.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Chiến lược không hỗ trợ: {strategy!r}. Chọn một trong {list(STRATEGIES)}")
    rng = np.random.default_rng(seed)
    main = main_presence(df, spec)
    weights = None
    if strategy == "rf":
        from .ml import rf_model
        weights = rf_model.next_draw_weights(main, spec, seed=seed)
    bonus_hist = bonus_presence(df, spec) if spec.bonus_player_picks else None
    return _generate_from(strategy, main, bonus_hist, spec, n_tickets, rng, window, weights)


# ---------------------------------------------------------------- backtest

def expected_matches(spec: GameSpec) -> tuple[float, float]:
    """(trung bình, độ lệch chuẩn) số trùng của một vé chọn ngẫu nhiên: phân phối siêu bội."""
    dist = stats.hypergeom(spec.max_number, spec.n_main, spec.n_main)
    return float(dist.mean()), float(dist.std())


def backtest(df: pd.DataFrame, spec: GameSpec, strategies: list[str] | None = None, n_draws: int = 200,
             tickets_per_draw: int = 5, window: int = 50, seed: int = 0,
             retrain_every: int = 20) -> pd.DataFrame:
    """Chạy lại lịch sử: với mỗi kỳ trong `n_draws` kỳ cuối, sinh vé chỉ từ dữ liệu TRƯỚC kỳ đó,
    rồi đếm số trùng với kết quả thật.

    Chiến lược "rf" huấn luyện lại Random Forest mỗi `retrain_every` kỳ (để backtest chạy nhanh);
    giữa các lần huấn luyện, đặc trưng vẫn được tính mới theo từng kỳ.

    Trả về DataFrame mỗi dòng một chiến lược: tickets, mean_matches, expected, z, p_value,
    hit_k (số vé trùng đúng k số, k = 0..n_main) và bonus_hits (nếu có).
    """
    strategies = strategies or list(STRATEGIES)
    main = main_presence(df, spec)
    has_b = spec.bonus_player_picks
    bonus_vals = df["bonus"].to_numpy(dtype=float) if has_b else None
    n_draws = min(n_draws, len(df) - 1)
    start = len(df) - n_draws
    mu, sigma = expected_matches(spec)
    features = None

    rows = []
    for si, strategy in enumerate(strategies):
        rng = np.random.default_rng([seed, si])
        hits = np.zeros(spec.n_main + 1, dtype=int)
        bonus_hits = 0
        total = 0
        model = None
        for t in range(start, len(df)):
            hist = main[:t]
            weights = None
            if strategy == "rf":
                from .ml import rf_model
                if features is None:
                    from .ml.features import build_features
                    features = build_features(main, spec)  # X[t] chỉ phụ thuộc main[:t]
                if model is None or (t - start) % retrain_every == 0:
                    model = rf_model.fit(features, main, end=t, seed=seed + si)  # chỉ dùng các kỳ < t
                weights = (np.ones(spec.max_number) if model is None
                           else rf_model.to_weights(rf_model.probabilities(model, features[t])))
            b_hist = None
            if has_b:
                past = bonus_vals[:t]
                past = past[~np.isnan(past)].astype(int)
                b_hist = np.zeros((len(past), spec.bonus_range_max), dtype=bool)
                b_hist[np.arange(len(past)), past - 1] = True
            actual = main[t]
            for tk in _generate_from(strategy, hist, b_hist, spec, tickets_per_draw, rng, window, weights):
                m = int(actual[np.array(tk.numbers) - 1].sum())
                hits[m] += 1
                total += 1
                if has_b and not np.isnan(bonus_vals[t]) and tk.bonus == int(bonus_vals[t]):
                    bonus_hits += 1
        mean = float((hits * np.arange(len(hits))).sum() / total)
        z = (mean - mu) / (sigma / np.sqrt(total))
        row = {"strategy": strategy, "tickets": total, "mean_matches": mean, "expected": mu,
               "z": z, "p_value": float(2 * stats.norm.sf(abs(z)))}
        row.update({f"hit_{k}": int(hits[k]) for k in range(len(hits))})
        if has_b:
            row["bonus_hits"] = bonus_hits
            row["bonus_expected"] = total / spec.bonus_range_max
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(n_draws=n_draws, tickets_per_draw=tickets_per_draw,
                     first_draw=int(df["draw_id"].iloc[start]), last_draw=int(df["draw_id"].iloc[-1]),
                     retrain_every=retrain_every)
    return out


def expected_hits(spec: GameSpec, tickets: int) -> dict[int, float]:
    """Số vé kỳ vọng trùng đúng k số khi chọn ngẫu nhiên `tickets` vé."""
    dist = stats.hypergeom(spec.max_number, spec.n_main, spec.n_main)
    return {k: float(dist.pmf(k) * tickets) for k in range(spec.n_main + 1)}
