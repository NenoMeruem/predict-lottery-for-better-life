"""
Random Forest dự đoán xác suất mỗi số xuất hiện ở kỳ tiếp theo.

Siêu tham số chọn theo hướng chống overfit (dữ liệu gần như nhiễu thuần):
cây nông, lá lớn, class_weight cân bằng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..games import GameSpec
from .features import FEATURE_NAMES, MIN_HISTORY, build_features, training_set

# Xác suất RF rất gần mức đều (~k/N) nên nếu dùng thẳng làm trọng số thì vé gần như ngẫu nhiên.
# Khuếch đại độ lệch tương đối: weight = (p / mean(p)) ** RF_SHARPEN.
RF_SHARPEN = 4.0

DEFAULT_PARAMS = dict(n_estimators=100, max_depth=6, min_samples_leaf=50, max_features="sqrt",
                      class_weight="balanced", n_jobs=-1)


def _forest(seed: int | None, params: dict | None):
    from sklearn.ensemble import RandomForestClassifier  # import muộn: chỉ cần khi dùng rf
    return RandomForestClassifier(random_state=seed, **{**DEFAULT_PARAMS, **(params or {})})


def fit(X: np.ndarray, P: np.ndarray, end: int, seed: int | None = 0, params: dict | None = None):
    """Huấn luyện trên các kỳ [MIN_HISTORY, end). Trả về model (hoặc None nếu chưa đủ dữ liệu)."""
    xs, ys = training_set(X, P, end)
    if len(np.unique(ys)) < 2 or end <= MIN_HISTORY:
        return None
    model = _forest(seed, params)
    model.fit(xs, ys)
    return model


def probabilities(model, X_t: np.ndarray) -> np.ndarray:
    """Xác suất mô hình gán cho mỗi số về ở kỳ có đặc trưng X_t (N x F). Chỉ có ý nghĩa để xếp hạng."""
    return model.predict_proba(X_t)[:, 1]


def to_weights(proba: np.ndarray, sharpen: float = RF_SHARPEN) -> np.ndarray:
    rel = proba / proba.mean()
    return np.maximum(rel, 1e-6) ** sharpen


def next_draw_weights(P: np.ndarray, spec: GameSpec, seed: int | None = 0,
                      params: dict | None = None) -> np.ndarray:
    """Huấn luyện trên toàn bộ lịch sử P và trả về trọng số chọn số cho kỳ kế tiếp."""
    X = build_features(P, spec)
    model = fit(X, P, end=len(P), seed=seed, params=params)
    if model is None:
        return np.ones(P.shape[1])
    return to_weights(probabilities(model, X[len(P)]))


def next_draw_probabilities(P: np.ndarray, spec: GameSpec, seed: int | None = 0,
                            params: dict | None = None) -> np.ndarray | None:
    X = build_features(P, spec)
    model = fit(X, P, end=len(P), seed=seed, params=params)
    return None if model is None else probabilities(model, X[len(P)])


def diagnostics(P: np.ndarray, spec: GameSpec, test_frac: float = 0.2, seed: int | None = 0,
                params: dict | None = None) -> dict:
    """Đánh giá ngoài mẫu: huấn luyện trên (1 - test_frac) đầu, đo AUC trên phần cuối.

    AUC ≈ 0.5 nghĩa là mô hình không phân biệt được số nào sẽ về tốt hơn đoán mò.
    Trả về dict: auc, base_rate, importances (DataFrame feature, importance), n_train, n_test.
    """
    from sklearn.metrics import roc_auc_score

    d = len(P)
    X = build_features(P, spec)
    split = int(d * (1 - test_frac))
    model = fit(X, P, end=split, seed=seed, params=params)
    xt = X[split:d].reshape(-1, X.shape[2])
    yt = P[split:d].reshape(-1).astype(int)
    auc = float(roc_auc_score(yt, probabilities(model, xt)))
    # Khoảng tin cậy xấp xỉ cho AUC ở mức "không có tín hiệu" (Hanley-McNeil): sd ≈ sqrt((n0+n1+1)/(12 n0 n1))
    n1 = int(yt.sum())
    n0 = len(yt) - n1
    sd = float(np.sqrt((n0 + n1 + 1) / (12 * n0 * n1)))
    imp = pd.DataFrame({"feature": FEATURE_NAMES, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return {"auc": auc, "auc_null_sd": sd, "base_rate": float(yt.mean()), "importances": imp,
            "n_train": int(split - MIN_HISTORY) * P.shape[1], "n_test": len(yt)}
