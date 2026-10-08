#!/usr/bin/env python3
"""
CLI phân tích kết quả Vietlott (535, 645, 655).

Lệnh con:
    report    làm sạch -> ghi dữ liệu sạch -> thống kê -> biểu đồ + báo cáo Markdown (mặc định)
    predict   sinh bộ số gợi ý cho kỳ tới theo các chiến lược thống kê
    backtest  chạy lại lịch sử để so độ trùng của từng chiến lược với chọn ngẫu nhiên

Ví dụ:
    python -m src.cli                              # = report cho mọi game có dữ liệu
    python -m src.cli 655 --window 100 --top 15    # = report 655 ...
    python -m src.cli report 645 --no-plots
    python -m src.cli predict 535                  # 7 chiến lược x 5 vé
    python -m src.cli predict 535 -s hot -s pairs -n 10 --seed 42
    python -m src.cli backtest 535 --draws 300 --tickets 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import analyzer, predictor
from .cleaner import clean, load_raw, raw_path, save_processed
from .games import GAMES, GameSpec, get_spec

COMMANDS = ("report", "predict", "backtest")
DISCLAIMER = ("Lưu ý: mỗi kỳ quay là ngẫu nhiên và độc lập. Các bộ số dưới đây KHÔNG làm tăng xác suất trúng; "
              "chạy `backtest` để kiểm chứng.")


def _fmt(nums) -> str:
    return " ".join(f"{int(n):02d}" for n in nums)


# ---------------------------------------------------------------- report

def print_summary(s: dict) -> None:
    fd, ld = s["first_draw"], s["last_draw"]
    print(f"  {s['draws']} kỳ: #{fd[0]:05d} ({fd[1]:%d/%m/%Y}) -> #{ld[0]:05d} ({ld[1]:%d/%m/%Y})")
    last = _fmt(s["last_numbers"]) + (f" | bonus {s['last_bonus']:02d}" if s["last_bonus"] else "")
    print(f"  Kỳ mới nhất     : {last}")
    print(f"  Về nhiều nhất   : {_fmt(s['most_frequent'])}")
    print(f"  Về ít nhất      : {_fmt(s['least_frequent'])}")
    print(f"  Nóng ({s['window']} kỳ)    : {_fmt(s['hot'])}")
    print(f"  Lạnh ({s['window']} kỳ)    : {_fmt(s['cold'])}")
    print("  Lâu chưa về     : " + ", ".join(f"{n:02d}({g} kỳ)" for n, g in s["longest_absent"]))
    print("  Cặp hay về      : " + ", ".join(f"{a:02d}-{b:02d}({c})" for a, b, c in s["top_pairs"]))
    u = s["uniformity"]
    print(f"  Chi-square      : χ²={u['chi2']:.1f}, p={u['p_value']:.3f} → {u['conclusion']}")


def write_index(out_dir: Path) -> Path:
    lines = ["# Báo cáo thống kê Vietlott\n"]
    for code, spec in GAMES.items():
        links = [f"[{name}]({code}/{f})" for name, f in (("thống kê", "README.md"), ("backtest", "backtest.md"))
                 if (out_dir / code / f).exists()]
        if links:
            lines.append(f"- **{spec.name}**: " + " · ".join(links))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cmd_report(args, spec: GameSpec) -> None:
    df, report = clean(load_raw(spec.code, args.data_dir), spec)
    print(report)
    for p in save_processed(df, spec.code, args.data_dir):
        print(f"  -> {p}")
    print_summary(analyzer.summary(df, spec, window=args.window, top=args.top))
    if not args.no_plots:
        from .visualizer import build_report  # import muộn: --no-plots không cần matplotlib
        md = build_report(df, spec, clean_report=report, out_dir=args.out_dir, window=args.window, top=args.top)
        print(f"  -> {md}")


# ---------------------------------------------------------------- predict / backtest

def _strategies(selected: list[str] | None) -> list[str]:
    if not selected or "all" in selected:
        return list(predictor.STRATEGIES)
    return list(dict.fromkeys(selected))


def _load(args, spec: GameSpec):
    df, report = clean(load_raw(spec.code, args.data_dir), spec)
    if report.dropped_rows:
        print(report)
    return df


def cmd_predict(args, spec: GameSpec) -> None:
    df = _load(args, spec)
    last = df.iloc[-1]
    print(f"  Dựa trên {len(df)} kỳ, mới nhất #{int(last['draw_id']):05d} ({last['date']:%d/%m/%Y}). "
          f"Gợi ý cho kỳ #{int(last['draw_id']) + 1:05d}:")
    if spec.bonus_player_picks:
        print(f"  (định dạng: {spec.n_main} số chính 1..{spec.max_number} | số đặc biệt 1..{spec.bonus_range_max})")
    for i, strategy in enumerate(_strategies(args.strategy)):
        seed = None if args.seed is None else args.seed + i
        tickets = predictor.generate(df, spec, strategy, n_tickets=args.tickets, window=args.window, seed=seed)
        print(f"\n  [{strategy}] {predictor.STRATEGIES[strategy]}")
        for j, t in enumerate(tickets, 1):
            print(f"    {j:>2}. {t}")


def cmd_backtest(args, spec: GameSpec) -> None:
    df = _load(args, spec)
    strategies = _strategies(args.strategy)
    print(f"  Backtest {min(args.draws, len(df) - 1)} kỳ cuối × {args.tickets} vé/kỳ × {len(strategies)} chiến lược...")
    bt = predictor.backtest(df, spec, strategies, n_draws=args.draws, tickets_per_draw=args.tickets,
                            window=args.window, seed=args.seed, retrain_every=args.retrain_every)
    mu, _ = predictor.expected_matches(spec)
    hit_cols = [f"hit_{k}" for k in range(spec.n_main + 1)]
    header = f"  {'chiến lược':<10} {'trùng TB':>8} {'p-value':>8}  " + " ".join(f"{'=' + c[4:]:>5}" for c in hit_cols)
    if "bonus_hits" in bt:
        header += f" {'bonus':>6}"
    print(header)
    for r in bt.itertuples():
        line = f"  {r.strategy:<10} {r.mean_matches:>8.3f} {r.p_value:>8.3f}  " + " ".join(
            f"{getattr(r, c):>5}" for c in hit_cols)
        if "bonus_hits" in bt:
            line += f" {r.bonus_hits:>6}"
        print(line)
    exp = predictor.expected_hits(spec, int(bt["tickets"].iloc[0]))
    line = f"  {'ngẫu nhiên':<10} {mu:>8.3f} {'':>8}  " + " ".join(f"{exp[k]:>5.1f}" for k in range(spec.n_main + 1))
    if "bonus_hits" in bt:
        line += f" {bt['bonus_expected'].iloc[0]:>6.1f}"
    print(line + "   <- kỳ vọng lý thuyết")
    rf_diag = None
    if "rf" in strategies:
        from .ml import rf_model
        rf_diag = rf_model.diagnostics(predictor.main_presence(df, spec), spec, seed=args.seed)
        z = (rf_diag["auc"] - 0.5) / rf_diag["auc_null_sd"]
        print(f"  RF ngoài mẫu: AUC={rf_diag['auc']:.4f} (đoán mò=0.5, z={z:+.2f}); "
              "đặc trưng quan trọng nhất: " + ", ".join(rf_diag["importances"]["feature"].head(3)))
    if not args.no_save:
        from .visualizer import build_backtest_report
        print(f"  -> {build_backtest_report(bt, spec, out_dir=args.out_dir, window=args.window, rf_diag=rf_diag)}")


# ---------------------------------------------------------------- argparse

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("games", nargs="*", help=f"mã sản phẩm ({', '.join(GAMES)}); mặc định: mọi game có dữ liệu")
    common.add_argument("-d", "--data-dir", default="data", help="thư mục chứa CSV thô (mặc định: data)")
    common.add_argument("-w", "--window", type=int, default=50, help="số kỳ gần nhất để xét nóng/lạnh (mặc định: 50)")

    strat = argparse.ArgumentParser(add_help=False)
    strat.add_argument("-s", "--strategy", action="append", choices=[*predictor.STRATEGIES, "all"],
                       help="chiến lược (lặp lại để chọn nhiều; mặc định: all). "
                            + "; ".join(f"{k}: {v}" for k, v in predictor.STRATEGIES.items()))

    ap = argparse.ArgumentParser(prog="python -m src.cli", description="Phân tích & gợi ý số Vietlott",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", parents=[common], help="thống kê + biểu đồ + báo cáo Markdown")
    p.add_argument("-o", "--out-dir", default="reports", help="thư mục báo cáo (mặc định: reports)")
    p.add_argument("-t", "--top", type=int, default=10, help="số lượng mục trong các bảng top")
    p.add_argument("--no-plots", action="store_true", help="không vẽ biểu đồ/báo cáo")

    p = sub.add_parser("predict", parents=[common, strat], help="sinh bộ số gợi ý cho kỳ tới")
    p.add_argument("-n", "--tickets", type=int, default=5, help="số vé mỗi chiến lược (mặc định: 5)")
    p.add_argument("--seed", type=int, default=None, help="seed để tái lập kết quả")

    p = sub.add_parser("backtest", parents=[common, strat], help="kiểm chứng chiến lược trên dữ liệu lịch sử")
    p.add_argument("--draws", type=int, default=200, help="số kỳ cuối dùng để backtest (mặc định: 200)")
    p.add_argument("-n", "--tickets", type=int, default=5, help="số vé mỗi kỳ mỗi chiến lược (mặc định: 5)")
    p.add_argument("--seed", type=int, default=0, help="seed (mặc định: 0)")
    p.add_argument("-o", "--out-dir", default="reports", help="thư mục báo cáo (mặc định: reports)")
    p.add_argument("--retrain-every", type=int, default=20,
                   help="chiến lược rf: huấn luyện lại sau mỗi N kỳ khi backtest (mặc định: 20)")
    p.add_argument("--no-save", action="store_true", help="không ghi backtest.md / backtest.png")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Tương thích ngược: `python -m src.cli 645 ...` = `python -m src.cli report 645 ...`
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, "report")
    args = build_parser().parse_args(argv)

    games = args.games or [g for g in GAMES if raw_path(g, args.data_dir).exists()]
    try:
        specs = [get_spec(g) for g in games]
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    missing = [s.code for s in specs if not raw_path(s.code, args.data_dir).exists()]
    if missing:
        print(f"Chưa có dữ liệu cho {missing}. Chạy: python3 src/vietlott_scraper.py {' '.join(missing)}",
              file=sys.stderr)
        return 2

    handler = {"report": cmd_report, "predict": cmd_predict, "backtest": cmd_backtest}[args.command]
    if args.command == "predict":
        print(DISCLAIMER)
    for spec in specs:
        print(f"\n=== {spec.name} ===")
        handler(args, spec)

    if args.command == "report" and not args.no_plots or args.command == "backtest" and not args.no_save:
        print(f"\nMục lục: {write_index(Path(args.out_dir))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
