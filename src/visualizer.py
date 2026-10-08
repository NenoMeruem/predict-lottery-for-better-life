"""
Vẽ biểu đồ từ kết quả của analyzer.py và tạo báo cáo Markdown có nhúng ảnh.

- Ảnh tĩnh (PNG) dùng Matplotlib + Seaborn, nhúng trực tiếp vào reports/<game>/README.md.
- Bản tương tác (HTML) dùng Plotly, được link từ file Markdown.

    from src.visualizer import build_report
    build_report(df, spec)   # -> reports/645/README.md
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # chạy được khi không có màn hình

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402

from . import analyzer, predictor  # noqa: E402
from .games import GameSpec  # noqa: E402

HOT, COLD, NEUTRAL, EXPECTED = "#d62728", "#1f77b4", "#9e9e9e", "#2ca02c"
DPI = 120

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------- helpers

def _save(fig: plt.Figure, out_path: str | Path | None) -> plt.Figure:
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return fig


def _number_ticks(ax: plt.Axes, fontsize: int = 8) -> None:
    ax.tick_params(axis="x", labelsize=fontsize, rotation=90)


# ---------------------------------------------------------------- static charts (Matplotlib / Seaborn)

def plot_frequency(freq: pd.DataFrame, spec: GameSpec, title: str | None = None,
                   out_path: str | Path | None = None) -> plt.Figure:
    """Bar tần suất từng số, tô màu trên/dưới kỳ vọng, kèm đường kỳ vọng."""
    data = freq.assign(status=np.where(freq["count"] >= freq["expected"], "Trên kỳ vọng", "Dưới kỳ vọng"))
    fig, ax = plt.subplots(figsize=(max(10, len(freq) * 0.28), 4.8))
    sns.barplot(data=data, x="number", y="count", hue="status", dodge=False, ax=ax,
                palette={"Trên kỳ vọng": HOT, "Dưới kỳ vọng": COLD})
    ax.axhline(freq["expected"].iloc[0], color=EXPECTED, ls="--", lw=1.5,
               label=f"Kỳ vọng ≈ {freq['expected'].iloc[0]:.1f}")
    lo, hi = freq["count"].min(), freq["count"].max()
    pad = max(5, (hi - lo) * 0.3)
    ax.set_ylim(max(0, lo - pad), hi + pad * 0.5)
    ax.set(xlabel="Số", ylabel="Số lần xuất hiện", title=title or f"{spec.name}: tần suất xuất hiện")
    ax.legend(loc="upper right", fontsize=9)
    _number_ticks(ax)
    return _save(fig, out_path)


def plot_hot_cold(hc: pd.DataFrame, spec: GameSpec, out_path: str | Path | None = None) -> plt.Figure:
    window = hc.attrs.get("window", "?")
    hot = hc[hc["status"] == "hot"].sort_values("recent_count")
    cold = hc[hc["status"] == "cold"].sort_values("recent_count", ascending=False)
    exp = hc["recent_expected"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 * max(len(hot), len(cold)) + 1.5), sharex=True)
    for ax, part, color, label in ((axes[0], hot, HOT, "Số nóng"), (axes[1], cold, COLD, "Số lạnh")):
        ax.barh(part["number"].astype(str), part["recent_count"], color=color)
        ax.axvline(exp, color=EXPECTED, ls="--", lw=1.5, label=f"Kỳ vọng ≈ {exp:.1f}")
        for y, v in enumerate(part["recent_count"]):
            ax.text(v + 0.1, y, str(v), va="center", fontsize=9)
        ax.set(title=label, xlabel=f"Số lần về trong {window} kỳ gần nhất", ylabel="Số")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{spec.name}: số nóng / lạnh ({window} kỳ gần nhất)")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_gaps(gp: pd.DataFrame, spec: GameSpec, out_path: str | Path | None = None) -> plt.Figure:
    """current_gap (cột) và avg_gap (chấm) của từng số."""
    exp = gp["expected_gap"].iloc[0]
    colors = np.where(gp["current_gap"] >= 2 * exp, HOT, NEUTRAL)
    fig, ax = plt.subplots(figsize=(max(10, spec.max_number * 0.28), 4.8))
    ax.bar(gp["number"].astype(str), gp["current_gap"], color=colors, label="Số kỳ chưa về hiện tại")
    ax.scatter(gp["number"].astype(str), gp["avg_gap"], color="black", s=14, zorder=3, label="Khoảng cách TB")
    ax.axhline(exp, color=EXPECTED, ls="--", lw=1.5, label=f"Kỳ vọng ≈ {exp:.1f}")
    ax.set(xlabel="Số", ylabel="Số kỳ", title=f"{spec.name}: số kỳ chưa về (đỏ = ≥ 2× kỳ vọng)")
    ax.set_ylim(0, max(gp["current_gap"].max(), gp["avg_gap"].max(skipna=True) or 0, exp) * 1.3)
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax.margins(x=0.01)
    _number_ticks(ax)
    return _save(fig, out_path)


def plot_pair_heatmap(pm: pd.DataFrame, spec: GameSpec, out_path: str | Path | None = None) -> plt.Figure:
    """Heatmap số lần đồng xuất hiện của mọi cặp (nửa dưới, bỏ đường chéo)."""
    mask = np.triu(np.ones_like(pm, dtype=bool))
    size = max(9, spec.max_number * 0.24)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(pm, mask=mask, cmap="rocket_r", square=True, ax=ax,
                xticklabels=True, yticklabels=True, cbar_kws={"label": "Số kỳ cùng về", "shrink": 0.7})
    ax.grid(False)
    ax.tick_params(labelsize=7)
    ax.set(xlabel="Số", ylabel="Số", title=f"{spec.name}: tần suất cặp số cùng về")
    return _save(fig, out_path)


def plot_top_pairs(pairs: pd.DataFrame, spec: GameSpec, out_path: str | Path | None = None) -> plt.Figure:
    data = pairs.iloc[::-1]
    labels = data["a"].astype(str) + " – " + data["b"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(data) + 1.5))
    ax.barh(labels, data["count"], color=HOT)
    ax.axvline(pairs["expected"].iloc[0], color=EXPECTED, ls="--", lw=1.5,
               label=f"Kỳ vọng ≈ {pairs['expected'].iloc[0]:.1f}")
    ax.set(xlabel="Số kỳ cùng về", ylabel="Cặp số", title=f"{spec.name}: top {len(pairs)} cặp số hay về cùng nhau")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, out_path)


def plot_frequency_over_time(by_period: pd.DataFrame, spec: GameSpec,
                             out_path: str | Path | None = None) -> plt.Figure:
    """Heatmap số x năm, tô màu quanh mức kỳ vọng (đỏ = cao hơn, xanh = thấp hơn)."""
    expected = spec.p_number * 100
    fig, ax = plt.subplots(figsize=(max(7, by_period.shape[1] * 0.9), spec.max_number * 0.2 + 1.5))
    sns.heatmap(by_period, cmap="RdBu_r", center=expected, ax=ax, yticklabels=True,
                cbar_kws={"label": "% số kỳ có số này"})
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    ax.set(xlabel="Năm", ylabel="Số", title=f"{spec.name}: tần suất theo năm (kỳ vọng ≈ {expected:.1f}%)")
    return _save(fig, out_path)


def plot_sum_distribution(df: pd.DataFrame, spec: GameSpec, out_path: str | Path | None = None) -> plt.Figure:
    st = analyzer.sum_stats(df, spec)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.histplot(df["sum"], bins=40, kde=True, color=COLD, ax=ax, stat="count")
    ax.axvline(st["mean"], color=HOT, lw=2, label=f"TB thực tế = {st['mean']:.1f}")
    ax.axvline(st["expected_mean"], color=EXPECTED, ls="--", lw=2, label=f"TB lý thuyết = {st['expected_mean']:.1f}")
    ax.set(xlabel=f"Tổng {spec.n_main} số chính", ylabel="Số kỳ", title=f"{spec.name}: phân phối tổng các số")
    ax.legend(fontsize=9)
    return _save(fig, out_path)


def plot_composition(odd: pd.DataFrame, low: pd.DataFrame, spec: GameSpec,
                     out_path: str | Path | None = None) -> plt.Figure:
    """So sánh tỉ lệ thực tế và lý thuyết của số lượng số lẻ / số nhỏ trong mỗi kỳ."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    half = spec.max_number // 2
    for ax, dist, label in ((axes[0], odd, "Số lượng số lẻ"), (axes[1], low, f"Số lượng số nhỏ (≤ {half})")):
        data = dist.melt(id_vars="k", value_vars=["pct", "expected_pct"], var_name="type", value_name="value")
        data["type"] = data["type"].map({"pct": "Thực tế", "expected_pct": "Lý thuyết"})
        sns.barplot(data=data, x="k", y="value", hue="type", ax=ax, palette={"Thực tế": COLD, "Lý thuyết": EXPECTED})
        ax.set(xlabel=label + " trong 1 kỳ", ylabel="% số kỳ")
        ax.legend(fontsize=9)
    fig.suptitle(f"{spec.name}: cơ cấu chẵn/lẻ và lớn/nhỏ")
    fig.tight_layout()
    return _save(fig, out_path)


# ---------------------------------------------------------------- interactive charts (Plotly)

def interactive_frequency(freq: pd.DataFrame, spec: GameSpec, out_path: str | Path,
                          title: str | None = None) -> go.Figure:
    data = freq.assign(status=np.where(freq["count"] >= freq["expected"], "Trên kỳ vọng", "Dưới kỳ vọng"))
    fig = px.bar(data, x="number", y="count", color="status",
                 color_discrete_map={"Trên kỳ vọng": HOT, "Dưới kỳ vọng": COLD},
                 hover_data={"pct": ":.2f", "expected": ":.1f", "deviation": ":+.1f"},
                 labels={"number": "Số", "count": "Số lần", "status": ""},
                 title=title or f"{spec.name}: tần suất xuất hiện")
    fig.add_hline(y=freq["expected"].iloc[0], line_dash="dash", line_color=EXPECTED, annotation_text="Kỳ vọng")
    fig.update_xaxes(dtick=1)
    _write_html(fig, out_path)
    return fig


def interactive_pair_heatmap(pm: pd.DataFrame, spec: GameSpec, out_path: str | Path) -> go.Figure:
    z = pm.to_numpy().astype(float)
    np.fill_diagonal(z, np.nan)
    fig = go.Figure(go.Heatmap(z=z, x=pm.columns, y=pm.index, colorscale="Reds",
                               hovertemplate="Cặp %{y} – %{x}: %{z} kỳ<extra></extra>",
                               colorbar={"title": "Số kỳ"}))
    fig.update_layout(title=f"{spec.name}: tần suất cặp số cùng về", width=850, height=800,
                      xaxis={"dtick": 1, "title": "Số"}, yaxis={"dtick": 1, "title": "Số", "autorange": "reversed"})
    _write_html(fig, out_path)
    return fig


def _write_html(fig: go.Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


# ---------------------------------------------------------------- markdown report

def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _balls(nums) -> str:
    return " ".join(f"`{int(n):02d}`" for n in nums)


def _fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y") if pd.notna(d) else "—"


def build_report(df: pd.DataFrame, spec: GameSpec, clean_report=None, out_dir: str | Path = "reports",
                 window: int = 50, top: int = 10) -> Path:
    """Vẽ toàn bộ biểu đồ vào <out_dir>/<game>/ và tạo README.md nhúng các ảnh đó."""
    out = Path(out_dir) / spec.code
    out.mkdir(parents=True, exist_ok=True)

    freq = analyzer.frequency(df, spec)
    hc = analyzer.hot_cold(df, spec, window=window, top=top)
    gp = analyzer.gaps(df, spec)
    pm = analyzer.pair_matrix(df, spec)
    pairs = analyzer.pair_frequency(df, spec, top=20)
    triplets = analyzer.triplet_frequency(df, spec, top=top)
    by_year = analyzer.frequency_by_period(df, spec, "year")
    odd = analyzer.composition_distribution(df, spec, "odd")
    low = analyzer.composition_distribution(df, spec, "low")
    sstats = analyzer.sum_stats(df, spec)
    uni = analyzer.uniformity_test(df, spec)

    plot_frequency(freq, spec, out_path=out / "frequency.png")
    plot_hot_cold(hc, spec, out_path=out / "hot_cold.png")
    plot_gaps(gp, spec, out_path=out / "gaps.png")
    plot_pair_heatmap(pm, spec, out_path=out / "pair_heatmap.png")
    plot_top_pairs(pairs, spec, out_path=out / "top_pairs.png")
    plot_frequency_over_time(by_year, spec, out_path=out / "frequency_by_year.png")
    plot_sum_distribution(df, spec, out_path=out / "sum_distribution.png")
    plot_composition(odd, low, spec, out_path=out / "composition.png")
    interactive_frequency(freq, spec, out / "frequency.html")
    interactive_pair_heatmap(pm, spec, out / "pair_heatmap.html")

    bonus_freq = None
    if spec.has_bonus:
        bonus_freq = analyzer.frequency(df, spec, source="bonus")
        plot_frequency(bonus_freq, spec, title=f"{spec.name}: tần suất số đặc biệt (bonus)",
                       out_path=out / "bonus_frequency.png")

    first, last = df.iloc[0], df.iloc[-1]
    w = hc.attrs["window"]
    md: list[str] = []
    add = md.append

    add(f"# Báo cáo thống kê {spec.name}\n")
    add(f"_Cập nhật {datetime.now():%d/%m/%Y %H:%M} · {len(df):,} kỳ, từ kỳ #{int(first['draw_id']):05d} "
        f"({_fmt_date(first['date'])}) đến kỳ #{int(last['draw_id']):05d} ({_fmt_date(last['date'])})_\n")
    add("> [!WARNING]\n> Mỗi kỳ quay là ngẫu nhiên và độc lập. Các số liệu dưới đây chỉ mô tả quá khứ, "
        "**không** làm tăng khả năng trúng ở kỳ sau.\n")

    add("## Tổng quan\n")
    last_row = [f"#{int(last['draw_id']):05d}", _fmt_date(last["date"]), _balls(last[spec.main_cols])]
    headers = ["Kỳ mới nhất", "Ngày", "Bộ số"]
    if spec.has_bonus:
        headers.append("Bonus")
        last_row.append(_balls([last["bonus"]]) if pd.notna(last["bonus"]) else "—")
    add(_md_table(headers, [last_row]) + "\n")
    if clean_report is not None:
        add(f"- Làm sạch dữ liệu: giữ **{clean_report.kept_rows}/{clean_report.total_rows}** dòng, "
            f"loại {clean_report.dropped_rows}"
            + (f", {clean_report.missing_bonus} kỳ thiếu bonus" if clean_report.missing_bonus else "") + ".")
    add(f"- Kiểm định chi-square tính phân phối đều: χ² = {uni['chi2']:.1f} (df = {uni['dof']}), "
        f"p-value = **{uni['p_value']:.3f}** → {uni['conclusion']}\n")

    add("## 1. Tần suất xuất hiện\n")
    add("![Tần suất xuất hiện](frequency.png)\n")
    most, least = freq.nlargest(top, "count"), freq.nsmallest(top, "count")
    add(_md_table(["#", "Về nhiều nhất", "Số lần", "% kỳ", "Về ít nhất", "Số lần", "% kỳ"],
                  [[i + 1, _balls([m.number]), m.count, f"{m.pct:.1f}%", _balls([l.number]), l.count, f"{l.pct:.1f}%"]
                   for i, (m, l) in enumerate(zip(most.itertuples(), least.itertuples()))]))
    add(f"\nKỳ vọng mỗi số ≈ **{freq['expected'].iloc[0]:.1f}** lần. [Bản tương tác (HTML)](frequency.html)\n")

    add(f"## 2. Số nóng / lạnh ({w} kỳ gần nhất)\n")
    add("![Số nóng lạnh](hot_cold.png)\n")
    hot = hc[hc["status"] == "hot"].sort_values("recent_count", ascending=False)
    cold = hc[hc["status"] == "cold"].sort_values("recent_count")
    add(f"- 🔥 **Nóng**: {_balls(hot['number'])}")
    add(f"- ❄️ **Lạnh**: {_balls(cold['number'])}")
    add(f"- Kỳ vọng mỗi số về ≈ {hc['recent_expected'].iloc[0]:.1f} lần trong {w} kỳ.\n")

    add("## 3. Số lâu chưa về\n")
    add("![Số kỳ chưa về](gaps.png)\n")
    absent = gp.nlargest(top, "current_gap")
    add(_md_table(["Số", "Số kỳ chưa về", "Lần về gần nhất", "Khoảng cách TB", "Khoảng cách dài nhất"],
                  [[_balls([r.number]), r.current_gap, f"#{int(r.last_seen_draw):05d} ({_fmt_date(r.last_seen_date)})",
                    f"{r.avg_gap:.1f}", r.max_gap] for r in absent.itertuples()]))
    add(f"\nKhoảng cách kỳ vọng ≈ **{gp['expected_gap'].iloc[0]:.1f}** kỳ.\n")

    add("## 4. Cặp số hay về cùng nhau\n")
    add("![Heatmap cặp số](pair_heatmap.png)\n")
    add("![Top cặp số](top_pairs.png)\n")
    add(f"Kỳ vọng mỗi cặp ≈ **{pairs['expected'].iloc[0]:.1f}** kỳ. "
        "[Heatmap tương tác (HTML)](pair_heatmap.html)\n")
    add(f"**Top {len(triplets)} bộ ba số hay về cùng nhau:**\n")
    add(_md_table(["#", "Bộ ba", "Số kỳ"],
                  [[i + 1, _balls([r.a, r.b, r.c]), r.count] for i, r in enumerate(triplets.itertuples())]) + "\n")

    add("## 5. Tần suất theo năm\n")
    add("![Tần suất theo năm](frequency_by_year.png)\n")

    add("## 6. Tổng các số\n")
    add("![Phân phối tổng](sum_distribution.png)\n")
    add(_md_table(["", "Trung bình", "Độ lệch chuẩn", "Min", "Max"],
                  [["Thực tế", f"{sstats['mean']:.1f}", f"{sstats['std']:.1f}", sstats["min"], sstats["max"]],
                   ["Lý thuyết", f"{sstats['expected_mean']:.1f}", f"{sstats['expected_std']:.1f}", "", ""]]) + "\n")

    add("## 7. Chẵn/lẻ, lớn/nhỏ\n")
    add("![Cơ cấu chẵn lẻ lớn nhỏ](composition.png)\n")

    if bonus_freq is not None:
        add("## 8. Số đặc biệt (bonus)\n")
        add("![Tần suất bonus](bonus_frequency.png)\n")
        bm, bl = bonus_freq.nlargest(5, "count"), bonus_freq.nsmallest(5, "count")
        add(f"- Về nhiều nhất: {_balls(bm['number'])}")
        add(f"- Về ít nhất: {_balls(bl['number'])}\n")

    path = out / "README.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


# ---------------------------------------------------------------- backtest report

def plot_backtest(bt: pd.DataFrame, spec: GameSpec, out_path: str | Path | None = None) -> plt.Figure:
    """Trái: số trùng TB mỗi vé (± khoảng tin cậy 95%) so với ngẫu nhiên.
    Phải: % vé trùng >= 3 số so với kỳ vọng."""
    mu, sigma = predictor.expected_matches(spec)
    ci = 1.96 * sigma / np.sqrt(bt["tickets"])
    alpha_adj = 0.05 / len(bt)
    colors = [HOT if p < alpha_adj else COLD for p in bt["p_value"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(bt) + 2.2), sharey=True)

    ax = axes[0]
    ax.barh(bt["strategy"], bt["mean_matches"], xerr=ci, color=colors, capsize=4)
    ax.axvline(mu, color=EXPECTED, ls="--", lw=1.5, label=f"Ngẫu nhiên = {mu:.3f}")
    span = max(ci.max() * 2.5, (bt["mean_matches"] - mu).abs().max() * 1.5)
    ax.set_xlim(mu - span, mu + span)
    ax.set(xlabel="Số trùng trung bình / vé (± 95% CI)", ylabel="Chiến lược")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()

    ax = axes[1]
    k3 = [c for c in bt.columns if c.startswith("hit_") and int(c[4:]) >= 3]
    rate = bt[k3].sum(axis=1) / bt["tickets"] * 100
    exp_rate = sum(v for k, v in predictor.expected_hits(spec, 1).items() if k >= 3) * 100
    ax.barh(bt["strategy"], rate, color=NEUTRAL)
    ax.axvline(exp_rate, color=EXPECTED, ls="--", lw=1.5, label=f"Kỳ vọng = {exp_rate:.2f}%")
    for y, v in enumerate(rate):
        ax.text(v, y, f" {v:.2f}%", va="center", fontsize=9)
    ax.set(xlabel="% vé trùng ≥ 3 số")
    ax.legend(loc="lower right", fontsize=9)

    a = bt.attrs
    fig.suptitle(f"{spec.name}: backtest {a.get('n_draws', '?')} kỳ × {a.get('tickets_per_draw', '?')} vé/kỳ "
                 f"(đỏ = lệch có ý nghĩa, p < {alpha_adj:.4f} sau hiệu chỉnh Bonferroni)")
    fig.tight_layout()
    return _save(fig, out_path)


def build_backtest_report(bt: pd.DataFrame, spec: GameSpec, out_dir: str | Path = "reports",
                          window: int = 50, rf_diag: dict | None = None) -> Path:
    """Ghi <out_dir>/<game>/backtest.md + backtest.png."""
    out = Path(out_dir) / spec.code
    out.mkdir(parents=True, exist_ok=True)
    plot_backtest(bt, spec, out_path=out / "backtest.png")
    a = bt.attrs
    mu, _ = predictor.expected_matches(spec)
    has_b = "bonus_hits" in bt.columns
    k = spec.n_main
    alpha_adj = 0.05 / len(bt)

    md = [f"# Backtest chiến lược chọn số {spec.name}\n",
          f"_Cập nhật {datetime.now():%d/%m/%Y %H:%M} · {a['n_draws']} kỳ (#{a['first_draw']:05d} → "
          f"#{a['last_draw']:05d}) · {a['tickets_per_draw']} vé/kỳ/chiến lược · cửa sổ nóng/lạnh {window} kỳ_\n",
          "> [!WARNING]\n> Mỗi vé chỉ được sinh từ dữ liệu **trước** kỳ đó (không nhìn trước kết quả). "
          "Nếu một chiến lược thực sự có lợi thế, số trùng trung bình phải cao hơn rõ rệt so với chọn ngẫu nhiên.\n",
          "![Backtest](backtest.png)\n",
          "## Chiến lược\n",
          _md_table(["Tên", "Mô tả"], [[f"`{s}`", d] for s, d in predictor.STRATEGIES.items() if s in set(bt["strategy"])]),
          "\n## Kết quả\n"]
    headers = ["Chiến lược", "Số vé", "Trùng TB", "p-value"] + [f"Trùng {i}" for i in range(k + 1)]
    if has_b:
        headers.append("Trúng bonus")
    exp_hits = predictor.expected_hits(spec, int(bt["tickets"].iloc[0]))
    rows = []
    for r in bt.itertuples():
        row = [f"`{r.strategy}`", r.tickets, f"{r.mean_matches:.3f}", f"{r.p_value:.3f}"]
        row += [getattr(r, f"hit_{i}") for i in range(k + 1)]
        if has_b:
            row.append(r.bonus_hits)
        rows.append(row)
    exp_row = ["_kỳ vọng (ngẫu nhiên)_", int(bt["tickets"].iloc[0]), f"{mu:.3f}", ""]
    exp_row += [f"{exp_hits[i]:.1f}" for i in range(k + 1)]
    if has_b:
        exp_row.append(f"{bt['bonus_expected'].iloc[0]:.1f}")
    rows.append(exp_row)
    md.append(_md_table(headers, rows))

    best = bt.loc[bt["mean_matches"].idxmax()]
    sig = bt[bt["p_value"] < alpha_adj]
    md.append("\n## Kết luận\n")
    md.append(f"- Chiến lược có số trùng TB cao nhất: `{best['strategy']}` ({best['mean_matches']:.3f} so với "
              f"{mu:.3f} của chọn ngẫu nhiên), p-value = {best['p_value']:.3f}.")
    md.append(f"- Vì so sánh {len(bt)} chiến lược cùng lúc, ngưỡng có ý nghĩa sau hiệu chỉnh Bonferroni là "
              f"p < {alpha_adj:.4f}. " + (
                  f"Có {len(sig)} chiến lược vượt ngưỡng: {', '.join(f'`{s}`' for s in sig['strategy'])}. "
                  "Nên chạy lại với `--seed` khác và nhiều kỳ hơn trước khi tin vào kết quả này."
                  if len(sig) else
                  "**Không chiến lược nào** vượt ngưỡng, nên chênh lệch quan sát được chỉ là dao động ngẫu nhiên."))
    if rf_diag is not None:
        plot_rf_importance(rf_diag["importances"], spec, out_path=out / "rf_importance.png")
        z_auc = (rf_diag["auc"] - 0.5) / rf_diag["auc_null_sd"]
        md.append("\n## Random Forest: đánh giá ngoài mẫu\n")
        md.append(f"Huấn luyện trên {rf_diag['n_train']:,} mẫu đầu, kiểm tra trên {rf_diag['n_test']:,} mẫu cuối "
                  f"(tỉ lệ số về = {rf_diag['base_rate']:.3f}).\n")
        md.append(f"- **AUC = {rf_diag['auc']:.4f}** (đoán mò = 0.5, độ lệch chuẩn khi không có tín hiệu "
                  f"≈ {rf_diag['auc_null_sd']:.4f}, tức z = {z_auc:+.2f}). "
                  + ("Mô hình không phân biệt được số nào sẽ về tốt hơn đoán mò."
                     if abs(z_auc) < 2 else
                     "Chênh lệch lớn hơn 2 độ lệch chuẩn, cần kiểm tra lại bằng dữ liệu/kỳ khác trước khi kết luận."))
        md.append("- Độ quan trọng của đặc trưng chỉ cho biết cây dùng đặc trưng nào để chia nhánh, "
                  "**không** chứng minh đặc trưng đó có sức dự báo (khi AUC ≈ 0.5, đó chỉ là khớp nhiễu).\n")
        md.append("![Độ quan trọng đặc trưng](rf_importance.png)\n")
    path = out / "backtest.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def plot_rf_importance(imp: pd.DataFrame, spec: GameSpec, out_path: str | Path | None = None) -> plt.Figure:
    data = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(data) + 1.5))
    ax.barh(data["feature"], data["importance"], color=COLD)
    ax.axvline(1 / len(imp), color=EXPECTED, ls="--", lw=1.5, label="Mức đều (1/số đặc trưng)")
    ax.set(xlabel="Độ quan trọng (Gini)", title=f"{spec.name}: độ quan trọng đặc trưng của Random Forest")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, out_path)
