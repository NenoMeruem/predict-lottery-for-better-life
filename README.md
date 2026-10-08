# Vietlott scraper

Lấy toàn bộ kết quả xổ số từ https://vietlott.vn/ ra CSV (mỗi sản phẩm một file). Chỉ cần Python 3.8+, không cài thêm thư viện.

```bash
python3 src/vietlott_scraper.py                  # tất cả sản phẩm -> ./data/*.csv
python3 src/vietlott_scraper.py 645 655          # chỉ Mega 6/45 và Power 6/55
python3 src/vietlott_scraper.py keno -w 8        # 8 request song song
python3 src/vietlott_scraper.py -o ketqua        # đổi thư mục lưu
```

| Mã | Sản phẩm | Cột |
|---|---|---|
| `645` | Mega 6/45 (từ 2016) | `n1..n6` |
| `655` | Power 6/55 (từ 2017) | `n1..n6`, `bonus` (số đặc biệt) |
| `535` | Lotto 5/35 (từ 2025, 2 kỳ/ngày) | `n1..n5`, `bonus` |
| `max3d` | Max 3D / Max 3D+ (từ 2019) | `giai_dac_biet`, `giai_nhat`, `giai_nhi`, `giai_ba` (các bộ số cách nhau bởi dấu cách) |
| `max3dpro` | Max 3D Pro (từ 2021) | như Max 3D |
| `keno` | Keno | `n1..n20`, `chan_le`, `lon_nho` |
| `bingo18` | Bingo18 | `n1..n3`, `tong`, `lon_hoa_nho` |

Mọi file đều có thêm `draw_id` (kỳ quay) và `date` (dd/mm/yyyy), sắp theo kỳ tăng dần.

**Chạy lại** cùng lệnh để chỉ lấy thêm các kỳ mới (vài giây). Nếu bị ngắt giữa chừng (Ctrl+C, mất mạng), chạy lại sẽ tiếp tục từ chỗ dừng (trạng thái lưu trong `data/.<mã>.state.json`).

**Lưu ý về Keno và Bingo18:** website chỉ còn giữ khoảng 8 tháng gần nhất cho Keno và khoảng 14 tháng cho Bingo18; các kỳ cũ hơn không còn trên website. Tool không bao giờ xoá dòng đã lưu, nên chạy định kỳ (ví dụ mỗi ngày bằng cron) sẽ giữ được toàn bộ lịch sử từ nay về sau. Lần chạy đầu tiên Keno có khoảng 4.800 trang và Bingo18 khoảng 11.500 trang, nên mất từ vài chục phút tới hơn 1 giờ.

## Phân tích thống kê & gợi ý số (Lotto 5/35, Mega 6/45, Power 6/55)

Cần cài thêm thư viện (pandas, seaborn, plotly...):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# report (mặc định): thống kê + biểu đồ + báo cáo Markdown
.venv/bin/python -m src.cli                       # mọi game có file data/<mã>.csv
.venv/bin/python -m src.cli 655 -w 100 -t 15      # nóng/lạnh theo 100 kỳ, bảng top 15
.venv/bin/python -m src.cli report 645 --no-plots # chỉ in thống kê ra terminal

# predict: sinh bộ số gợi ý cho kỳ tới
.venv/bin/python -m src.cli predict 535                         # 7 chiến lược x 5 vé
.venv/bin/python -m src.cli predict 535 -s hot -s pairs -n 10   # chọn chiến lược, 10 vé
.venv/bin/python -m src.cli predict 535 --seed 42               # tái lập kết quả

# backtest: so độ trùng của từng chiến lược với chọn ngẫu nhiên trên lịch sử
.venv/bin/python -m src.cli backtest 535 --draws 300 -n 10      # -> reports/535/backtest.md

.venv/bin/python -m pytest -q                     # chạy tests
```

Chiến lược (`-s`): `random`, `frequency`, `hot`, `cold`, `overdue`, `pairs`, `balanced`, `rf` (Random Forest), hoặc `all` (mặc định).
Với Lotto 5/35, mỗi vé gồm 5 số (1–35) và 1 số đặc biệt (1–12).

| Module | Vai trò |
|---|---|
| `src/cleaner.py` | Chuẩn hoá kiểu dữ liệu, loại dòng thiếu/sai (có ghi lý do), ghi `data/processed/<mã>.parquet` + `.csv` |
| `src/analyzer.py` | Tần suất, số nóng/lạnh, số lâu chưa về, cặp/bộ ba hay về, tần suất theo năm, chẵn/lẻ, lớn/nhỏ, kiểm định chi-square |
| `src/visualizer.py` | Biểu đồ PNG (Matplotlib/Seaborn) nhúng vào báo cáo Markdown, kèm bản tương tác HTML (Plotly) |
| `src/predictor.py` | Sinh bộ số theo chiến lược + backtest walk-forward (chỉ dùng dữ liệu trước mỗi kỳ) |
| `src/ml/` | `features.py` (đặc trưng không rò rỉ tương lai) và `rf_model.py` (Random Forest + AUC ngoài mẫu) |
| `src/cli.py` | Lệnh `report` / `predict` / `backtest` |

Báo cáo: [`reports/README.md`](reports/README.md).

> Mỗi kỳ quay là ngẫu nhiên và độc lập. Các thống kê chỉ mô tả quá khứ; `backtest` cho thấy không chiến lược nào trùng nhiều hơn chọn ngẫu nhiên một cách có ý nghĩa.
