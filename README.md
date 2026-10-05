# Vietlott scraper

Lấy toàn bộ kết quả xổ số từ https://vietlott.vn/ ra CSV (mỗi sản phẩm một file). Chỉ cần Python 3.8+, không cài thêm thư viện.

```bash
python3 vietlott_scraper.py                  # tất cả sản phẩm -> ./data/*.csv
python3 vietlott_scraper.py 645 655          # chỉ Mega 6/45 và Power 6/55
python3 vietlott_scraper.py keno -w 8        # 8 request song song
python3 vietlott_scraper.py -o ketqua        # đổi thư mục lưu
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
