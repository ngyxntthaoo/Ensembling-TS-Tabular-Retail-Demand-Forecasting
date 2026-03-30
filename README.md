# Inventory Forecasting — SuperC

Dự báo tồn kho hàng ngày cho chuỗi bán lẻ sử dụng nhiều phương pháp từ baseline thống kê đến deep learning với entity embedding.

---

## Bài toán

Dự báo **số lượng sản phẩm bán ra (Units Sold)** cho **100 SKU** (5 cửa hàng × 20 sản phẩm) theo các horizon **7, 14, 28 ngày**.

- **Dữ liệu**: `Model/dataset/sales_data.csv` — chuỗi thời gian hàng ngày từ 2022-01-01 đến 2024-01-30
- **Train**: đến 2023-06-30 | **Val**: đến 2023-10-31 | **Test**: từ 2023-11-01 (rolling evaluation)
- **Đánh giá**: sMAPE, MASE, RMSE, RMSLE trên rolling window

---

## Cấu trúc project

```
├── Model/
│   ├── dataset/
│   │   └── sales_data.csv
│   ├── baseline/               # Các model baseline
│   │   ├── metrics.py          # Hàm tính sMAPE, MASE, RMSE, RMSLE
│   │   ├── run_naive.py
│   │   ├── run_snaive.py
│   │   ├── run_mnaive.py
│   │   ├── run_arima.py
│   │   ├── run_ets.py
│   │   ├── run_sarimax.py
│   │   ├── run_prophet.ipynb
│   │   └── run_chronos.ipynb
│   ├── proposed/               # Model đề xuất: LSTM + Entity Embedding
│   │   ├── lstm-multi-entity-embedding.ipynb          # Model chính
│   │   ├── lstm-multi-entity-embedding-tuning-with-optuna.ipynb
│   │   ├── lstm-multi-entity-embedding-tuning-with-gao.ipynb
│   │   ├── lstm-multi-entity-embedding-tuning-with-hbo.ipynb
│   │   ├── ablation-A1-no-categorical.ipynb
│   │   ├── ablation-A2-onehot.ipynb
│   │   ├── ablation-A3-ordinal.ipynb
│   │   ├── ablation-A4-no-external.ipynb
│   │   └── feature-importance.ipynb
│   └── result/                 # Kết quả đã chạy (CSV + PNG)
├── PAPER/                      # Tài liệu tham khảo
├── dashboard.py                # Streamlit dashboard
├── requirement.txt
└── README.md
```

---

## Các model

| Nhóm | Model | File |
|------|-------|------|
| Baseline | Naive, SNaive, MNaive | `run_naive/snaive/mnaive.py` |
| Statistical | ARIMA, ETS, SARIMAX | `run_arima/ets/sarimax.py` |
| ML/DL | Prophet, Chronos-Bolt-Small, LSTM-Univariate | `run_prophet/chronos.ipynb`, `run_lstm_uni.py` |
| **Proposed** | **LSTM + Entity Embedding** | `lstm-multi-entity-embedding.ipynb` |

**Model đề xuất** sử dụng entity embedding để học đặc trưng riêng của từng cửa hàng và sản phẩm, kết hợp với calendar features và external features, được tuning bằng Optuna / GAO / HBO.

---

## Kết quả (mean sMAPE, horizon=7)

| Model | sMAPE (%) | MASE |
|-------|-----------|------|
| Chronos-Bolt-Small | **29.77** | **0.71** |
| LSTM-EntityEmb (Optuna) | ~37–38 | ~0.75 |
| LSTM-Univariate | 38.92 | 0.88 |
| MNaive | 39.71 | 0.85 |
| ARIMA | 39.93 | 0.85 |
| Naive | 48.41 | 1.04 |

> MASE < 1 nghĩa là model tốt hơn Naive forecast. Xem kết quả đầy đủ trong `Model/result/` hoặc chạy dashboard.

---

## Cài đặt

```bash
pip install -r requirement.txt
pip install torch prophet statsmodels streamlit plotly
```

> Chronos yêu cầu thêm: `pip install chronos-forecasting`  
> Prophet yêu cầu: `pip install prophet`

---

## Cách chạy

### 1. Chạy baseline models

```bash
cd Model/baseline

python run_naive.py
python run_snaive.py
python run_mnaive.py
python run_arima.py
python run_ets.py
python run_sarimax.py
```

Kết quả lưu vào `Model/result/*_summary.csv` và `*_details.csv`.

Với Prophet và Chronos, mở notebook tương ứng:
```
Model/baseline/run_prophet.ipynb
Model/baseline/run_chronos.ipynb
```

### 2. Chạy model đề xuất

Mở và chạy theo thứ tự:

```
Model/proposed/lstm-multi-entity-embedding.ipynb        # train model chính
Model/proposed/lstm-multi-entity-embedding-tuning-with-optuna.ipynb  # tuning
```

### 3. Chạy ablation study

```
Model/proposed/ablation-A1-no-categorical.ipynb   # bỏ entity embedding
Model/proposed/ablation-A2-onehot.ipynb           # thay bằng one-hot
Model/proposed/ablation-A3-ordinal.ipynb          # thay bằng ordinal
Model/proposed/ablation-A4-no-external.ipynb      # bỏ external features
```

### 4. Xem kết quả qua dashboard

```bash
cd /path/to/Inventory-forecasting---SuperC
python3 -m streamlit run dashboard.py
```

Dashboard gồm 4 tab:
- **Model Comparison** — so sánh tất cả models theo metric và horizon
- **Ablation Study** — phân tích đóng góp từng thành phần
- **Per-SKU Analysis** — phân tích lỗi theo từng SKU
- **Plots** — visualize actual vs predicted

---

## Metrics

| Metric | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| **sMAPE** | Sai số % đối xứng, không phụ thuộc scale | Càng thấp càng tốt |
| **MASE** | So sánh với Naive forecast | < 1 (tốt hơn Naive) |
| **RMSE** | Sai số tuyệt đối, phạt nặng outlier | Càng thấp càng tốt |
| **RMSLE** | RMSE trên log scale, robust với outlier | Càng thấp càng tốt |

---

## Tài liệu tham khảo

- Chronos: *Learning the Language of Time Series* — [arxiv](https://arxiv.org/pdf/2403.07815) | [github](https://github.com/amazon-science/chronos-forecasting)
- [FinTSB benchmark](https://github.com/TongjiFinLab/FinTSB) — XGBoost + Chronos
- [Prophet](https://github.com/imnileshd/time-series-prophet.git)
