import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

RESULT_DIR = Path("Model/result")

st.set_page_config(page_title="Inventory Forecasting Dashboard", layout="wide")
st.title("📦 Inventory Forecasting — EDA & Results")

# ── Metric descriptions ───────────────────────────────────────────────────────

METRIC_INFO = {
    "mean_smape": {
        "label": "sMAPE (%)",
        "desc": "**Symmetric Mean Absolute Percentage Error (sMAPE)** — đo lường sai số theo phần trăm, đối xứng giữa giá trị thực và dự báo. "
                "Giá trị 0% là hoàn hảo; thường dưới 20% được coi là tốt trong bài toán bán lẻ. "
                "Ưu điểm: không bị ảnh hưởng bởi scale của dữ liệu, dễ so sánh giữa các SKU khác nhau.",
        "good": "↓ càng thấp càng tốt",
    },
    "median_smape": {
        "label": "Median sMAPE (%)",
        "desc": "Trung vị của sMAPE trên toàn bộ SKU — ít bị ảnh hưởng bởi các SKU outlier so với mean. "
                "Nên dùng khi phân phối lỗi lệch (skewed), ví dụ một vài SKU có sai số rất lớn.",
        "good": "↓ càng thấp càng tốt",
    },
    "mean_mase": {
        "label": "MASE",
        "desc": "**Mean Absolute Scaled Error (MASE)** — so sánh sai số của model với sai số của Naive forecast (dự báo = giá trị ngày hôm trước). "
                "MASE < 1: model tốt hơn Naive. MASE = 1: ngang bằng Naive. MASE > 1: tệ hơn Naive. "
                "Đây là metric chính để đánh giá vì nó có ý nghĩa tương đối và không bị ảnh hưởng bởi scale.",
        "good": "↓ < 1 là tốt hơn baseline Naive",
    },
    "median_mase": {
        "label": "Median MASE",
        "desc": "Trung vị của MASE — robust hơn mean khi có outlier. Nếu median_mase < 1, hơn 50% số SKU được dự báo tốt hơn Naive.",
        "good": "↓ < 1 là tốt hơn baseline Naive",
    },
    "mean_rmse": {
        "label": "RMSE",
        "desc": "**Root Mean Squared Error** — đo sai số tuyệt đối theo cùng đơn vị với dữ liệu (số lượng sản phẩm). "
                "Phạt nặng các sai số lớn hơn MAE. Hữu ích khi muốn tránh các dự báo lệch lớn.",
        "good": "↓ càng thấp càng tốt (đơn vị: số lượng SP)",
    },
    "mean_rmsle": {
        "label": "RMSLE",
        "desc": "**Root Mean Squared Log Error** — tương tự RMSE nhưng tính trên log scale, phù hợp khi dữ liệu có phân phối lệch phải "
                "hoặc khi sai số tương đối quan trọng hơn sai số tuyệt đối. Ít nhạy cảm với outlier hơn RMSE.",
        "good": "↓ càng thấp càng tốt",
    },
}

ABLATION_INFO = {
    "A1-NoCategorical": "Bỏ toàn bộ categorical features (store, product embedding) — kiểm tra tầm quan trọng của entity embedding.",
    "A2-OneHot":        "Thay entity embedding bằng one-hot encoding — so sánh hai cách biểu diễn categorical.",
    "A3-Ordinal":       "Thay entity embedding bằng ordinal encoding — cách đơn giản nhất để encode categorical.",
    "A4-NoExternal":    "Bỏ external features (calendar, promotion, v.v.) — kiểm tra đóng góp của feature bên ngoài.",
    "A5-NoCalendar":    "Bỏ riêng calendar features (day of week, month, holiday) — kiểm tra tầm quan trọng của thông tin thời gian.",
}

# ── Load helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def load_summaries():
    frames = []
    for f in sorted(RESULT_DIR.glob("*_summary.csv")):
        if f.name.startswith("ablation"):
            continue
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)

@st.cache_data
def load_ablation():
    frames = []
    for f in sorted(RESULT_DIR.glob("ablation_*.csv")):
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)

@st.cache_data
def load_details():
    frames = []
    for f in sorted(RESULT_DIR.glob("*_detail*.csv")):
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Comparison", "🔬 Ablation Study", "🏪 Per-SKU Analysis", "🖼️ Plots"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    ### So sánh các model dự báo

    Bài toán dự báo tồn kho trên **100 SKU** (10 cửa hàng × 10 sản phẩm), với 3 horizon: **7, 14, 28 ngày**.
    Dữ liệu train đến `2023-06-30`, test từ `2023-11-01` trở đi theo rolling evaluation.

    **Các model được so sánh:**
    - **Naive / SNaive / MNaive** — baseline đơn giản, dự báo = giá trị ngày trước / cùng ngày tuần trước / trung bình
    - **ARIMA / ETS / SARIMAX** — mô hình thống kê cổ điển
    - **Prophet** — mô hình additive của Meta, xử lý tốt seasonality và holiday
    - **Chronos-Bolt** — foundation model của Amazon, zero-shot forecasting
    - **LSTM-EntityEmbedding** — mô hình đề xuất, dùng entity embedding để học đặc trưng riêng của từng SKU
    """)

    raw = load_summaries()
    available_metrics = [c for c in ["mean_smape", "median_smape", "mean_mase", "median_mase",
                                      "mean_rmse", "mean_rmsle"] if c in raw.columns]
    df = raw.dropna(subset=available_metrics, how="all")

    col1, col2 = st.columns(2)
    horizon = col1.selectbox("Horizon (days)", sorted(df["horizon"].unique()), key="h1")
    metric  = col2.selectbox("Metric", available_metrics, key="met1",
                              format_func=lambda x: METRIC_INFO.get(x, {}).get("label", x))

    # Metric explanation box
    if metric in METRIC_INFO:
        info = METRIC_INFO[metric]
        st.info(f"**{info['label']}** — {info['desc']}  \n🎯 *{info['good']}*")

    sub = df[df["horizon"] == horizon].dropna(subset=[metric]).sort_values(metric)

    fig = px.bar(
        sub, x="model", y=metric, color="model",
        title=f"{METRIC_INFO.get(metric, {}).get('label', metric)} theo từng model (horizon = {horizon} ngày)",
        text=sub[metric].round(4),
        labels={metric: METRIC_INFO.get(metric, {}).get("label", metric)},
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    if not sub.empty:
        best_model = sub.iloc[0]["model"]
        best_val   = sub.iloc[0][metric]
        naive_row  = sub[sub["model"].str.lower().str.contains("naive")]
        naive_val  = naive_row.iloc[0][metric] if not naive_row.empty else None

        interp = f"✅ **Model tốt nhất** tại horizon={horizon}: **{best_model}** với {METRIC_INFO.get(metric,{}).get('label',metric)} = **{best_val:.4f}**"
        if naive_val and metric in ["mean_mase", "median_mase"]:
            improvement = (1 - best_val / naive_val) * 100
            interp += f"\n\nSo với Naive ({naive_val:.4f}), cải thiện **{improvement:.1f}%**."
        st.success(interp)

    st.subheader("Heatmap — tất cả horizons")
    st.caption("Màu xanh = tốt hơn (giá trị thấp hơn). Dễ thấy model nào ổn định qua các horizon.")
    pivot = df.dropna(subset=[metric]).pivot_table(index="model", columns="horizon", values=metric)
    fig2 = px.imshow(
        pivot, text_auto=".3f", color_continuous_scale="RdYlGn_r",
        title=f"{METRIC_INFO.get(metric,{}).get('label', metric)} — heatmap theo horizon",
        labels={"color": METRIC_INFO.get(metric,{}).get("label", metric)},
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Xem bảng số liệu đầy đủ"):
        st.dataframe(df.sort_values(["horizon", metric]), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Ablation Study
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    ### Ablation Study — LSTM Entity Embedding

    Mục tiêu: kiểm tra đóng góp của từng thành phần trong model đề xuất bằng cách **loại bỏ từng phần** và đo sự thay đổi performance.
    Nếu loại bỏ một thành phần làm kết quả xấu đi đáng kể → thành phần đó quan trọng.
    """)

    abl = load_ablation()
    abl_metrics = [c for c in ["mean_smape", "median_smape", "mean_mase", "median_mase",
                                "mean_rmse", "mean_rmsle"] if c in abl.columns]

    col1, col2 = st.columns(2)
    horizon_a = col1.selectbox("Horizon", sorted(abl["horizon"].unique()), key="h2")
    metric_a  = col2.selectbox("Metric", abl_metrics, key="m2",
                                format_func=lambda x: METRIC_INFO.get(x, {}).get("label", x))

    if metric_a in METRIC_INFO:
        st.info(f"**{METRIC_INFO[metric_a]['label']}** — {METRIC_INFO[metric_a]['desc']}")

    sub_a = abl[abl["horizon"] == horizon_a].dropna(subset=[metric_a]).sort_values(metric_a).copy()

    # Ablation descriptions
    st.markdown("**Ý nghĩa các variant:**")
    for k, v in ABLATION_INFO.items():
        if any(k in str(a) for a in sub_a["ablation"].values):
            st.markdown(f"- `{k}`: {v}")

    fig3 = px.bar(
        sub_a, x="ablation", y=metric_a, color="ablation",
        title=f"Ablation: {METRIC_INFO.get(metric_a,{}).get('label', metric_a)} @ horizon={horizon_a} ngày",
        text=sub_a[metric_a].round(4),
        labels={metric_a: METRIC_INFO.get(metric_a,{}).get("label", metric_a)},
    )
    fig3.update_layout(showlegend=False, xaxis_tickangle=-20)
    st.plotly_chart(fig3, use_container_width=True)

    best_val = sub_a[metric_a].min()
    sub_a["delta_vs_best"] = (sub_a[metric_a] - best_val).round(4)
    sub_a["delta_%"] = ((sub_a[metric_a] - best_val) / best_val * 100).round(2)

    st.caption("**delta_vs_best**: chênh lệch so với variant tốt nhất. Variant nào có delta lớn → thành phần bị loại bỏ quan trọng.")
    st.dataframe(
        sub_a[["ablation", "model", "horizon", metric_a, "delta_vs_best", "delta_%"]],
        use_container_width=True
    )

    if not sub_a.empty:
        worst = sub_a.iloc[-1]
        st.warning(f"⚠️ Variant **{worst['ablation']}** có {METRIC_INFO.get(metric_a,{}).get('label',metric_a)} cao nhất "
                   f"({worst[metric_a]:.4f}), tệ hơn best **{worst['delta_%']:.1f}%** → thành phần bị loại bỏ có đóng góp lớn nhất.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Per-SKU Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ### Phân tích theo từng SKU

    Không phải tất cả SKU đều được dự báo tốt như nhau. Tab này giúp xác định:
    - **SKU nào khó dự báo nhất** (lỗi cao) → cần xem xét riêng
    - **Phân phối lỗi** có đều không hay bị lệch bởi một vài SKU outlier
    - **Model nào phù hợp nhất** cho từng SKU cụ thể
    """)

    det = load_details()
    det.columns = [c.lower() for c in det.columns]

    det_metrics = [c for c in ["smape", "mase", "rmse", "rmsle"] if c in det.columns]

    col1, col2, col3 = st.columns(3)
    sel_model = col1.selectbox("Model", sorted(det["model"].unique()), key="m3")
    sel_h     = col2.selectbox("Horizon", sorted(det["horizon"].unique()), key="h3")
    metric_d  = col3.selectbox("Metric", det_metrics, key="md3",
                                format_func=lambda x: METRIC_INFO.get("mean_"+x, {}).get("label", x.upper()))

    full_metric_key = "mean_" + metric_d
    if full_metric_key in METRIC_INFO:
        st.info(f"**{METRIC_INFO[full_metric_key]['label']}** — {METRIC_INFO[full_metric_key]['desc']}")

    sub_d = det[(det["model"] == sel_model) & (det["horizon"] == sel_h)].copy()

    if "store" in sub_d.columns and "product" in sub_d.columns:
        sub_d["sku"] = sub_d["store"] + " | " + sub_d["product"]
    else:
        sub_d["sku"] = sub_d.get("product", sub_d.index.astype(str))

    sub_d = sub_d.sort_values(metric_d, ascending=False)

    # Summary stats
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Mean " + metric_d.upper(), f"{sub_d[metric_d].mean():.4f}")
    col_b.metric("Median " + metric_d.upper(), f"{sub_d[metric_d].median():.4f}")
    col_c.metric("Worst SKU", f"{sub_d[metric_d].max():.4f}")
    col_d.metric("Best SKU", f"{sub_d[metric_d].min():.4f}")

    if metric_d == "mase":
        pct_better = (sub_d["mase"] < 1).mean() * 100
        if pct_better >= 50:
            st.success(f"✅ **{pct_better:.0f}%** SKU có MASE < 1 → model tốt hơn Naive trên đa số SKU.")
        else:
            st.warning(f"⚠️ Chỉ **{pct_better:.0f}%** SKU có MASE < 1 → model chưa vượt được Naive trên phần lớn SKU.")

    st.subheader("Phân phối lỗi")
    st.caption("Histogram cho thấy lỗi tập trung ở đâu. Đuôi dài bên phải = có một số SKU rất khó dự báo.")
    fig4 = px.histogram(sub_d, x=metric_d, nbins=40,
                        title=f"Phân phối {metric_d.upper()} — {sel_model} @ horizon={sel_h}",
                        labels={metric_d: metric_d.upper()})
    if metric_d == "mase":
        fig4.add_vline(x=1.0, line_dash="dash", line_color="red",
                       annotation_text="MASE=1 (ngang Naive)", annotation_position="top right")
    st.plotly_chart(fig4, use_container_width=True)

    top_n = st.slider("Top N SKU có lỗi cao nhất", 5, 30, 10)
    st.caption(f"Top {top_n} SKU khó dự báo nhất — cần xem xét thêm dữ liệu hoặc model riêng cho các SKU này.")
    fig5 = px.bar(
        sub_d.head(top_n), x="sku", y=metric_d,
        title=f"Top {top_n} SKU tệ nhất — {metric_d.upper()}",
        color=metric_d, color_continuous_scale="Reds",
        labels={metric_d: metric_d.upper()},
    )
    fig5.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("So sánh các model trên một SKU cụ thể")
    st.caption("Chọn một SKU để xem model nào dự báo tốt nhất cho SKU đó.")
    sel_sku = st.selectbox("Chọn SKU", sorted(sub_d["sku"].unique()))

    if "store" in det.columns:
        store_val, prod_val = sel_sku.split(" | ")
        sku_df = det[(det["store"] == store_val) & (det["product"] == prod_val) & (det["horizon"] == sel_h)]
    else:
        sku_df = det[(det["product"] == sel_sku) & (det["horizon"] == sel_h)]

    if not sku_df.empty and metric_d in sku_df.columns:
        sku_df_sorted = sku_df.sort_values(metric_d)
        best_m = sku_df_sorted.iloc[0]["model"]
        st.info(f"Model tốt nhất cho **{sel_sku}** tại horizon={sel_h}: **{best_m}** ({metric_d.upper()} = {sku_df_sorted.iloc[0][metric_d]:.4f})")
        fig6 = px.bar(
            sku_df_sorted, x="model", y=metric_d,
            title=f"{metric_d.upper()} cho {sel_sku} @ horizon={sel_h}",
            color=metric_d, color_continuous_scale="Blues_r",
            text=sku_df_sorted[metric_d].round(3),
            labels={metric_d: metric_d.upper()},
        )
        if metric_d == "mase":
            fig6.add_hline(y=1.0, line_dash="dash", line_color="red",
                           annotation_text="Naive baseline")
        st.plotly_chart(fig6, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Plots
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    ### Visualization dự báo

    Các biểu đồ dưới đây hiển thị **actual vs predicted** cho một số SKU đại diện,
    giúp trực quan hóa chất lượng dự báo và phát hiện các pattern mà metric số không thể hiện được
    (ví dụ: model bắt được trend nhưng lệch phase, hoặc dự báo tốt ở mùa thấp nhưng kém ở peak).
    """)

    plots = list(RESULT_DIR.glob("plot_*.png"))
    if plots:
        plot_labels = {
            "plot_chronos": "Chronos-Bolt — zero-shot foundation model",
            "plot_prophet": "Prophet — additive seasonality model",
            "plot_stat_models": "Statistical models (ARIMA, ETS, SARIMAX, Naive)",
        }
        cols = st.columns(min(len(plots), 2))
        for i, p in enumerate(sorted(plots)):
            label = plot_labels.get(p.stem, p.stem)
            cols[i % 2].image(str(p), caption=label, use_container_width=True)
    else:
        st.info("Chưa có file plot_*.png trong thư mục result.")
