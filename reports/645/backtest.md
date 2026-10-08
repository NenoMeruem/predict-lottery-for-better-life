# Backtest chiến lược chọn số Mega 6/45

_Cập nhật 08/10/2026 09:14 · 200 kỳ (#01372 → #01571) · 5 vé/kỳ/chiến lược · cửa sổ nóng/lạnh 50 kỳ_

> [!WARNING]
> Mỗi vé chỉ được sinh từ dữ liệu **trước** kỳ đó (không nhìn trước kết quả). Nếu một chiến lược thực sự có lợi thế, số trùng trung bình phải cao hơn rõ rệt so với chọn ngẫu nhiên.

![Backtest](backtest.png)

## Chiến lược

| Tên | Mô tả |
|---|---|
| `random` | Ngẫu nhiên đều (mốc so sánh) |
| `pairs` | Chọn dần theo cặp số hay về cùng nhau |
| `rf` | Random Forest (đặc trưng: độ trễ, tần suất trượt, cặp số, kỳ trước) |

## Kết quả

| Chiến lược | Số vé | Trùng TB | p-value | Trùng 0 | Trùng 1 | Trùng 2 | Trùng 3 | Trùng 4 | Trùng 5 | Trùng 6 |
|---|---|---|---|---|---|---|---|---|---|---|
| `random` | 1000 | 0.844 | 0.076 | 364 | 454 | 160 | 18 | 4 | 0 | 0 |
| `rf` | 1000 | 0.820 | 0.420 | 378 | 443 | 161 | 17 | 1 | 0 | 0 |
| `pairs` | 1000 | 0.776 | 0.333 | 415 | 412 | 156 | 16 | 1 | 0 | 0 |
| _kỳ vọng (ngẫu nhiên)_ | 1000 | 0.800 |  | 400.6 | 424.1 | 151.5 | 22.4 | 1.4 | 0.0 | 0.0 |

## Kết luận

- Chiến lược có số trùng TB cao nhất: `random` (0.844 so với 0.800 của chọn ngẫu nhiên), p-value = 0.076.
- Vì so sánh 3 chiến lược cùng lúc, ngưỡng có ý nghĩa sau hiệu chỉnh Bonferroni là p < 0.0167. **Không chiến lược nào** vượt ngưỡng, nên chênh lệch quan sát được chỉ là dao động ngẫu nhiên.

## Random Forest: đánh giá ngoài mẫu

Huấn luyện trên 55,170 mẫu đầu, kiểm tra trên 14,175 mẫu cuối (tỉ lệ số về = 0.133).

- **AUC = 0.5018** (đoán mò = 0.5, độ lệch chuẩn khi không có tín hiệu ≈ 0.0071, tức z = +0.25). Mô hình không phân biệt được số nào sẽ về tốt hơn đoán mò.
- Độ quan trọng của đặc trưng chỉ cho biết cây dùng đặc trưng nào để chia nhánh, **không** chứng minh đặc trưng đó có sức dự báo (khi AUC ≈ 0.5, đó chỉ là khớp nhiễu).

![Độ quan trọng đặc trưng](rf_importance.png)

