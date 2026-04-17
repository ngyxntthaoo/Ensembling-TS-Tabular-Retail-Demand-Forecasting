import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# Use __file__ so paths work regardless of cwd when streamlit is launched
_HERE       = Path(__file__).parent.resolve()
RESULT_DIR  = _HERE / "result"
DATA_PATH   = _HERE / "dataset" / "sales_data.csv"
TRAIN_END   = pd.Timestamp("2023-06-30")
VAL_END     = pd.Timestamp("2023-10-31")
HORIZON     = 7

def _to_rgba(color: str, alpha: float = 0.15) -> str:
    """Convert any CSS color (hex or rgb/rgba) to rgba() string for Plotly fillcolor."""
    if color.startswith("rgb"):
        return color.replace("rgb(", "rgba(").rstrip(")") + f", {alpha})"
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

MODEL_COLORS = {
    # ── Proposed / Ensemble ───────────────────────────────────────────────────
    "chronos_hgb_ensemble_hpo":   "#1b5e20",   # best proposed (HPO)
    "ensemble_tpe":               "#2e7d32",
    "ensemble_rand":              "#66bb6a",
    "chronos_hgb_ensemble":       "#43a047",   # baseline blend (no HPO)
    # ── Components ────────────────────────────────────────────────────────────
    "chronos_bolt_small":         "#5c85d6",
    "lgbm_chronos":               "#6a1b9a",
    "hgb_lag_features":           "#fb8c00",
    "ridge_lag_features":         "#ef6c00",
    "lstm_univariate":            "#880e4f",
    # ── Statistical ───────────────────────────────────────────────────────────
    "prophet":                    "#ff8f00",
    "arima":                      "#42a5f5",
    "sarimax":                    "#1565c0",
    "ets":                        "#0d47a1",
    # ── Baselines ─────────────────────────────────────────────────────────────
    "naive":                      "#90a4ae",
    "seasonal_naive":             "#546e7a",
}

PROPOSED = {
    "chronos_hgb_ensemble_hpo", "ensemble_tpe", "ensemble_rand",
    "chronos_hgb_ensemble", "chronos_bolt_small", "lgbm_chronos",
    "hgb_lag_features",
}

st.set_page_config(page_title="Inventory Forecasting Dashboard",
                   layout="wide", page_icon="📦")
st.title("📦 Inventory Forecasting — H = 7 days")
st.caption("Proposed model: **Chronos-Bolt-Small + HGB Ensemble** with HPO  ·  "
           "100 SKU (5 stores × 20 products)  ·  "
           f"Train → {TRAIN_END.date()}  |  Val → {VAL_END.date()}  |  Test → 2024-01-30")

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_sales():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    return df

@st.cache_data
def load_test_table():
    f = RESULT_DIR / "test_table_h7_timeseries.csv"
    if f.exists():
        return pd.read_csv(f)
    # fallback: results_h7.json
    fj = RESULT_DIR / "results_h7.json"
    if fj.exists():
        import json
        with open(fj) as fp:
            d = json.load(fp)
        return pd.DataFrame(d.get("test_table", []))
    return pd.DataFrame()

@st.cache_data
def load_val_table():
    fj = RESULT_DIR / "results_h7.json"
    if fj.exists():
        import json
        with open(fj) as fp:
            d = json.load(fp)
        return pd.DataFrame(d.get("val_table", []))
    return pd.DataFrame()

@st.cache_data
def load_feature_importance():
    # Priority 1: dedicated CSV (has std column too)
    f_csv = RESULT_DIR / "timeseries_h7_feature_importance.csv"
    if f_csv.exists() and f_csv.stat().st_size > 50:
        return pd.read_csv(f_csv)
    # Priority 2: embedded in results_h7.json
    fj = RESULT_DIR / "results_h7.json"
    if fj.exists():
        import json
        with open(fj) as fp:
            d = json.load(fp)
        rows = d.get("top_features_hpo", [])
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame()

@st.cache_data
def load_details():
    """Load per-series test metrics for all available models.

    Priority:
    1. timeseries_h7_details.csv — consolidated file with all models (preferred)
    2. Fall back to individual *_test_details.csv raw files
    Columns normalised to: store, product, test_MAE, test_RMSE, test_sMAPE, model
    """
    # ── 1. Prefer the consolidated details file (if non-empty) ───────────────
    consolidated = RESULT_DIR / "timeseries_h7_details.csv"
    if consolidated.exists() and consolidated.stat().st_size > 200:
        df = pd.read_csv(consolidated)
        # Normalise column names: mae → test_MAE, rmse → test_RMSE, smape → test_sMAPE
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "mae":          rename[c] = "test_MAE"
            elif cl == "rmse":       rename[c] = "test_RMSE"
            elif cl == "smape":      rename[c] = "test_sMAPE"
            elif cl == "test_mae":   rename[c] = "test_MAE"
            elif cl == "test_rmse":  rename[c] = "test_RMSE"
            elif cl == "test_smape": rename[c] = "test_sMAPE"
        return df.rename(columns=rename)

    # ── 2. Fall back: read individual raw *_test_details.csv files ────────────
    MODEL_MAP = {
        "chronos": "chronos_bolt_small",
        "ensemble": "chronos_hgb_ensemble",
    }
    frames = []
    for f in sorted(RESULT_DIR.glob("*_details.csv")):
        if f.name.startswith("timeseries_"):
            continue          # skip consolidated files to avoid duplication
        df = pd.read_csv(f)
        stem = f.stem.lower()
        model_name = next((v for k, v in MODEL_MAP.items() if k in stem), stem)
        if "model" not in [c.lower() for c in df.columns]:
            df["model"] = model_name
        # Normalise column names
        rename = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "test_mae":   rename[c] = "test_MAE"
            elif cl == "test_rmse":  rename[c] = "test_RMSE"
            elif cl == "test_smape": rename[c] = "test_sMAPE"
        df = df.rename(columns=rename)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_data
def load_ablation():
    """Load ablation study results from ablation_proposed_b1b6_summary.csv."""
    f = RESULT_DIR / "ablation_proposed_b1b6_summary.csv"
    if f.exists():
        return pd.read_csv(f)
    return pd.DataFrame()

# ── Seasonal naive helper ─────────────────────────────────────────────────────
def seasonal_naive_forecast(history: np.ndarray, steps: int, season: int = 7) -> np.ndarray:
    """Repeat the last full season for `steps` steps."""
    out = []
    for i in range(steps):
        out.append(history[-(season - (i % season))])
    return np.array(out)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Model Comparison",
    "📈 Actual vs Predicted",
    "🏪 Per-SKU Metrics",
    "🔬 Feature Importance",
    "🧪 Ablation Study",
    "ℹ️ About",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("So sánh tất cả models — Test Set")

    test_tbl = load_test_table()
    val_tbl  = load_val_table()

    if test_tbl.empty:
        st.warning("Chưa có file kết quả. Hãy chạy notebook để export.")
        st.stop()

    METRIC_COLS = {"test_MAE": "MAE", "test_RMSE": "RMSE", "test_sMAPE": "sMAPE (%)"}
    available = [c for c in METRIC_COLS if c in test_tbl.columns]

    col1, col2 = st.columns([2, 1])
    metric_sel = col1.selectbox("Metric", available,
                                format_func=lambda x: METRIC_COLS[x], key="t1_met")
    split_sel  = col2.selectbox("Split", ["test", "val"] if not val_tbl.empty else ["test"], key="t1_split")

    tbl = test_tbl if split_sel == "test" else val_tbl
    m_col = metric_sel if split_sel == "test" else metric_sel.replace("test_", "val_")

    if m_col not in tbl.columns:
        st.warning(f"Không có cột {m_col} trong bảng {split_sel}.")
    else:
        tbl_sorted = tbl.dropna(subset=[m_col]).sort_values(m_col).reset_index(drop=True)
        tbl_sorted["_prop"] = tbl_sorted["model"].isin(PROPOSED)
        tbl_sorted["_color"] = tbl_sorted["model"].map(
            lambda m: MODEL_COLORS.get(m, "#9e9e9e"))

        fig = go.Figure()
        for _, row in tbl_sorted.iterrows():
            fig.add_trace(go.Bar(
                x=[row["model"]],
                y=[row[m_col]],
                name=row["model"],
                marker_color=row["_color"],
                text=[f"{row[m_col]:.3f}"],
                textposition="outside",
                showlegend=False,
            ))
        best = tbl_sorted.iloc[0]
        fig.add_annotation(
            x=best["model"], y=best[m_col],
            text=f"🏆 Best<br>{best[m_col]:.3f}",
            showarrow=True, arrowhead=2, arrowcolor="#2e7d32",
            font=dict(color="#2e7d32", size=12), ay=-40,
        )
        fig.update_layout(
            title=f"{METRIC_COLS[metric_sel]} — {split_sel} set  (H=7d)",
            xaxis_tickangle=-30,
            yaxis_title=METRIC_COLS[metric_sel],
            height=420,
            margin=dict(t=50, b=10),
            plot_bgcolor="white",
        )
        fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

    # Best vs Naive improvement
    naive_row = tbl_sorted[tbl_sorted["model"].str.lower().str.contains("naive")]
    if not naive_row.empty:
        naive_val = naive_row.iloc[0][m_col]
        best_val  = best[m_col]
        imp = (1 - best_val / naive_val) * 100
        st.success(
            f"**{best['model']}** đạt {METRIC_COLS[metric_sel]} = **{best_val:.3f}**, "
            f"cải thiện **{imp:.1f}%** so với Naive ({naive_val:.3f})."
        )

        # Val vs Test comparison scatter
        if not val_tbl.empty:
            st.subheader("Val vs Test MAE — Generalization Check")
            st.caption("Điểm gần đường chéo = model không bị overfit; điểm trên đường chéo = overfit.")
            merged = pd.merge(
                val_tbl[["model", "val_MAE"]],
                test_tbl[["model", "test_MAE"]],
                on="model", how="inner"
            ).dropna()
            merged["_color"] = merged["model"].map(lambda m: MODEL_COLORS.get(m, "#9e9e9e"))
            merged["_prop"]  = merged["model"].isin(PROPOSED)

            fig_sc = go.Figure()
            lo = min(merged["val_MAE"].min(), merged["test_MAE"].min()) * 0.92
            hi = max(merged["val_MAE"].max(), merged["test_MAE"].max()) * 1.05
            fig_sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                        line=dict(color="#bdbdbd", dash="dash", width=1.5),
                                        name="Val = Test"))
            for _, row in merged.iterrows():
                fig_sc.add_trace(go.Scatter(
                    x=[row["val_MAE"]], y=[row["test_MAE"]],
                    mode="markers+text",
                    marker=dict(size=12 if row["_prop"] else 9,
                                color=row["_color"],
                                symbol="star" if row["_prop"] else "circle"),
                    text=[row["model"]], textposition="top right",
                    textfont=dict(size=9),
                    name=row["model"], showlegend=False,
                ))
            fig_sc.update_layout(
                xaxis_title="Val MAE", yaxis_title="Test MAE",
                height=380, plot_bgcolor="white",
            )
            fig_sc.update_xaxes(range=[lo, hi], gridcolor="#f0f0f0")
            fig_sc.update_yaxes(range=[lo, hi], gridcolor="#f0f0f0")
            st.plotly_chart(fig_sc, use_container_width=True)

        with st.expander("📋 Bảng số liệu đầy đủ"):
            st.dataframe(tbl_sorted.drop(columns=["_prop","_color"]), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Actual vs Predicted (Interactive time series)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Actual vs Predicted — Interactive Time Series")
    st.caption(
        "Chọn một SKU để xem toàn bộ time series, dự báo baseline (tính live), "
        "và error band cho proposed model (từ per-series MAE đã lưu)."
    )

    sales = load_sales()
    details = load_details()

    if sales.empty:
        st.error(f"Không tìm thấy file dữ liệu tại `{DATA_PATH}`. "
                 "Hãy chạy dashboard từ thư mục `Model/`.")
        st.stop()

    stores   = sorted(sales["Store ID"].unique())
    products = sorted(sales["Product ID"].unique())

    # ── SKU selector ─────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    sel_store   = c1.selectbox("Store", stores, key="t2_store")
    sel_product = c2.selectbox("Product", products, key="t2_product")

    sku_df = (sales[(sales["Store ID"] == sel_store) &
                    (sales["Product ID"] == sel_product)]
              .sort_values("Date").reset_index(drop=True))

    if sku_df.empty:
        st.warning("Không có dữ liệu cho SKU này.")
        st.stop()

    # ── Display controls ──────────────────────────────────────────────────────
    c3, c4, c5 = st.columns(3)
    show_train  = c3.toggle("Hiển thị Train period", value=True, key="t2_train")
    show_naive  = c4.toggle("Naive (computed live)", value=True, key="t2_naive")
    show_snaive = c5.toggle("Seasonal Naive (lag-7)", value=True, key="t2_snaive")

    # ── Model selector for error bands ────────────────────────────────────────
    _test_tbl_t2 = load_test_table()
    _avail_models = _test_tbl_t2["model"].tolist() if not _test_tbl_t2.empty else []
    _default_sel  = [m for m in ["chronos_hgb_ensemble", "chronos_bolt_small"] if m in _avail_models]
    sel_models_t2 = st.multiselect(
        "📊 Chọn models để hiển thị error band ±MAE trên Test set",
        options=_avail_models,
        default=_default_sel,
        format_func=lambda m: m.replace("_", " ").title(),
        key="t2_models",
    )
    _global_mae_map = (dict(zip(_test_tbl_t2["model"], _test_tbl_t2["test_MAE"]))
                       if not _test_tbl_t2.empty else {})

    # ── Split the series ──────────────────────────────────────────────────────
    train_s = sku_df[sku_df["Date"] <= TRAIN_END]
    val_s   = sku_df[(sku_df["Date"] > TRAIN_END) & (sku_df["Date"] <= VAL_END)]
    test_s  = sku_df[sku_df["Date"] > VAL_END]

    # ── Live baseline predictions for val + test ──────────────────────────────
    train_vals = train_s["Units Sold"].values

    # Naive: last training value repeated
    last_train  = float(train_vals[-1])
    naive_val   = np.full(len(val_s),  last_train)
    naive_test  = np.full(len(test_s), last_train)

    # Seasonal Naive: lag-7 rolling
    all_history = train_vals.copy()
    snaive_val, snaive_test = [], []
    for actual in val_s["Units Sold"].values:
        pred = all_history[-7] if len(all_history) >= 7 else all_history[-1]
        snaive_val.append(pred)
        all_history = np.append(all_history, actual)
    for actual in test_s["Units Sold"].values:
        pred = all_history[-7] if len(all_history) >= 7 else all_history[-1]
        snaive_test.append(pred)
        all_history = np.append(all_history, actual)
    snaive_val  = np.array(snaive_val)
    snaive_test = np.array(snaive_test)

    # ── Per-series MAE for proposed models ────────────────────────────────────
    sku_metrics = {}
    if not details.empty:
        # Find store/product columns case-insensitively
        col_lower = {c.lower(): c for c in details.columns}
        store_col   = col_lower.get("store", "store")
        product_col = col_lower.get("product", "product")
        model_col   = col_lower.get("model", "model")
        sku_det = details[
            (details[store_col]   == sel_store) &
            (details[product_col] == sel_product)
        ]
        for _, row in sku_det.iterrows():
            sku_metrics[row[model_col]] = {
                "mae":   row.get("test_MAE",   row.get("mae",   np.nan)),
                "rmse":  row.get("test_RMSE",  row.get("rmse",  np.nan)),
                "smape": row.get("test_sMAPE", row.get("smape", np.nan)),
            }

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Train actual (optional)
    if show_train and not train_s.empty:
        fig.add_trace(go.Scatter(
            x=train_s["Date"], y=train_s["Units Sold"],
            name="Actual (Train)", mode="lines",
            line=dict(color="#90a4ae", width=1.2),
            opacity=0.6,
        ))

    # Val actual
    if not val_s.empty:
        fig.add_trace(go.Scatter(
            x=val_s["Date"], y=val_s["Units Sold"],
            name="Actual (Val)", mode="lines",
            line=dict(color="#1565c0", width=2),
        ))

    # Test actual
    if not test_s.empty:
        fig.add_trace(go.Scatter(
            x=test_s["Date"], y=test_s["Units Sold"],
            name="Actual (Test)", mode="lines",
            line=dict(color="#212121", width=2.5),
        ))

    # Naive predictions
    if show_naive:
        if not val_s.empty:
            fig.add_trace(go.Scatter(
                x=val_s["Date"], y=naive_val,
                name="Naive (val)", mode="lines",
                line=dict(color="#90a4ae", width=1.5, dash="dot"),
                showlegend=False,
            ))
        if not test_s.empty:
            fig.add_trace(go.Scatter(
                x=test_s["Date"], y=naive_test,
                name="Naive", mode="lines",
                line=dict(color="#90a4ae", width=1.8, dash="dot"),
            ))

    # Seasonal Naive predictions
    if show_snaive:
        if not val_s.empty:
            fig.add_trace(go.Scatter(
                x=val_s["Date"], y=snaive_val,
                name="Seasonal Naive (val)", mode="lines",
                line=dict(color="#546e7a", width=1.5, dash="dash"),
                showlegend=False,
            ))
        if not test_s.empty:
            fig.add_trace(go.Scatter(
                x=test_s["Date"], y=snaive_test,
                name="Seasonal Naive", mode="lines",
                line=dict(color="#546e7a", width=1.8, dash="dash"),
            ))

    # Error bands — all selected models; per-series MAE preferred, global MAE as fallback
    if not test_s.empty and sel_models_t2:
        actual_arr = test_s["Units Sold"].values
        for model_name in sel_models_t2:
            # Prefer per-series MAE from details CSV, fall back to global test MAE
            per_sku_mae = sku_metrics.get(model_name, {}).get("mae", np.nan)
            mae = per_sku_mae if not np.isnan(per_sku_mae) else _global_mae_map.get(model_name, np.nan)
            if np.isnan(mae):
                continue
            source = "per-SKU" if not np.isnan(per_sku_mae) else "global avg"
            color = MODEL_COLORS.get(model_name, "#9e9e9e")
            upper = actual_arr + mae
            lower = np.maximum(0, actual_arr - mae)
            label = model_name.replace("_", " ").title()

            # Upper bound line
            fig.add_trace(go.Scatter(
                x=test_s["Date"], y=upper,
                mode="lines",
                line=dict(color=color, width=0.8, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
            # Lower bound line
            fig.add_trace(go.Scatter(
                x=test_s["Date"], y=lower,
                mode="lines",
                line=dict(color=color, width=0.8, dash="dot"),
                name=f"{label} (±MAE={mae:.1f}, {source})",
            ))

    # ── Split markers ─────────────────────────────────────────────────────────
    # add_vline + annotation separately to avoid Plotly bug with string x + annotation
    for date, label, color in [
        (TRAIN_END, "Train end", "#ef9a9a"),
        (VAL_END,   "Val end",   "#90caf9"),
    ]:
        fig.add_vline(x=str(date), line_dash="dash", line_color=color, line_width=1.5)
        fig.add_annotation(x=str(date), y=1, yref="paper",
                           text=label, showarrow=False,
                           font=dict(color=color), xanchor="left", yanchor="top")

    fig.update_layout(
        title=f"Units Sold — {sel_store} / {sel_product}  "
              f"(Train={len(train_s)}d · Val={len(val_s)}d · Test={len(test_s)}d)",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        height=480,
        hovermode="x unified",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="#f5f5f5", rangeslider=dict(visible=True))
    fig.update_yaxes(gridcolor="#f5f5f5")
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    if not test_s.empty:
        st.subheader("Test period metrics for this SKU")
        kpi_cols = st.columns(max(1, len(sku_metrics)) + 2)

        # Naive MAE (computed)
        naive_mae = float(np.mean(np.abs(naive_test - test_s["Units Sold"].values)))
        kpi_cols[0].metric("Naive MAE", f"{naive_mae:.2f}")

        # Snaive MAE
        snaive_mae = float(np.mean(np.abs(snaive_test - test_s["Units Sold"].values)))
        kpi_cols[1].metric("Seasonal Naive MAE", f"{snaive_mae:.2f}")

        for i, (mname, mets) in enumerate(sku_metrics.items()):
            mae = mets["mae"]
            delta = f"{((mae - naive_mae)/naive_mae*100):+.1f}% vs Naive" if not np.isnan(mae) else ""
            kpi_cols[i + 2].metric(
                mname.replace("_", " ").title(),
                f"{mae:.2f}" if not np.isnan(mae) else "N/A",
                delta=delta,
                delta_color="inverse",
            )

    # ── Rolling 7-day window breakdown ───────────────────────────────────────
    if not test_s.empty and len(test_s) >= HORIZON:
        with st.expander("📅 Rolling H=7 window breakdown — Naive vs Seasonal Naive"):
            windows = []
            n_windows = len(test_s) // HORIZON
            for w in range(n_windows):
                start = w * HORIZON
                end   = start + HORIZON
                win   = test_s.iloc[start:end]
                n_pred  = naive_test[start:end]
                sn_pred = snaive_test[start:end]
                actual  = win["Units Sold"].values
                windows.append({
                    "Window": w + 1,
                    "Start":  win["Date"].iloc[0].date(),
                    "End":    win["Date"].iloc[-1].date(),
                    "Actual Mean": round(actual.mean(), 2),
                    "Naive MAE":   round(np.mean(np.abs(n_pred - actual)), 2),
                    "SNaive MAE":  round(np.mean(np.abs(sn_pred - actual)), 2),
                })
            st.dataframe(pd.DataFrame(windows), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Per-SKU Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Per-SKU Test Metrics")

    details_all = load_details()
    if details_all.empty:
        st.warning("Chưa có file *_details.csv trong thư mục result.")
        st.stop()

    details_all.columns = [c.lower() for c in details_all.columns]
    det_metric_cols = [c for c in ["test_mae", "test_rmse", "test_smape"] if c in details_all.columns]
    model_list = sorted(details_all["model"].unique())

    c1, c2 = st.columns(2)
    sel_m   = c1.selectbox("Model", model_list, key="t3_model")
    sel_met = c2.selectbox("Metric", det_metric_cols,
                           format_func=lambda x: x.replace("test_", "").upper(),
                           key="t3_met")

    sub_d = details_all[details_all["model"] == sel_m].dropna(subset=[sel_met]).copy()
    if "store" in sub_d.columns and "product" in sub_d.columns:
        sub_d["sku"] = sub_d["store"] + " | " + sub_d["product"]
    sub_d = sub_d.sort_values(sel_met, ascending=False).reset_index(drop=True)

    # KPIs
    ka, kb, kc, kd = st.columns(4)
    ka.metric("Mean", f"{sub_d[sel_met].mean():.3f}")
    kb.metric("Median", f"{sub_d[sel_met].median():.3f}")
    kc.metric("Worst SKU", f"{sub_d[sel_met].max():.3f}")
    kd.metric("Best SKU", f"{sub_d[sel_met].min():.3f}")

    # Histogram
    fig_h = px.histogram(
        sub_d, x=sel_met, nbins=30,
        title=f"Distribution of {sel_met.upper()} — {sel_m}",
        labels={sel_met: sel_met.replace("test_", "").upper()},
        color_discrete_sequence=[MODEL_COLORS.get(sel_m, "#2e7d32")],
    )
    fig_h.update_layout(plot_bgcolor="white", height=300)
    st.plotly_chart(fig_h, use_container_width=True)

    # Top-N worst SKUs
    top_n = st.slider("Top N worst SKUs", 5, min(50, len(sub_d)), 20, key="t3_topn")
    worst = sub_d.head(top_n)
    fig_w = px.bar(
        worst, x="sku" if "sku" in worst.columns else "product",
        y=sel_met,
        color=sel_met, color_continuous_scale="Reds",
        title=f"Top {top_n} SKUs — highest {sel_met.upper()}",
        text=worst[sel_met].round(2),
        labels={sel_met: sel_met.replace("test_", "").upper()},
    )
    fig_w.update_layout(xaxis_tickangle=-40, plot_bgcolor="white", height=360)
    st.plotly_chart(fig_w, use_container_width=True)

    # Cross-model comparison for a single SKU
    st.subheader("Compare all models for one SKU")
    if "sku" in sub_d.columns:
        sku_list = sorted(sub_d["sku"].unique())
        sel_sku2 = st.selectbox("SKU", sku_list, key="t3_sku")
        s_val, p_val = sel_sku2.split(" | ")
        sku_cmp = details_all[
            (details_all["store"] == s_val) &
            (details_all["product"] == p_val)
        ].dropna(subset=[sel_met]).sort_values(sel_met).reset_index(drop=True)
        sku_cmp["_color"] = sku_cmp["model"].map(lambda m: MODEL_COLORS.get(m, "#9e9e9e"))

        fig_cmp = go.Figure()
        for _, row in sku_cmp.iterrows():
            fig_cmp.add_trace(go.Bar(
                x=[row["model"]], y=[row[sel_met]],
                marker_color=row["_color"],
                text=[f"{row[sel_met]:.3f}"], textposition="outside",
                name=row["model"], showlegend=False,
            ))
        fig_cmp.update_layout(
            title=f"{sel_met.upper()} for {sel_sku2}",
            xaxis_tickangle=-30, plot_bgcolor="white", height=350,
            yaxis_title=sel_met.replace("test_", "").upper(),
        )
        fig_cmp.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig_cmp, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Feature Importance
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Feature Importance — HGB component (Permutation Importance, Val Set)")
    st.caption("Giá trị = mức tăng RMSE khi xáo trộn ngẫu nhiên feature đó (10 lần lặp). "
               "Feature quan trọng → loại bỏ nó làm RMSE tăng nhiều.")

    fi = load_feature_importance()
    if fi.empty:
        st.warning("Chưa có dữ liệu feature importance. Hãy chạy notebook (cell 52) để export.")
    else:
        top_n_fi = st.slider("Top N features", 5, min(50, len(fi)), 20, key="t4_n")
        fi_sorted = fi.sort_values("importance", ascending=False).head(top_n_fi)

        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(
            x=fi_sorted["importance"],
            y=fi_sorted["feature"],
            orientation="h",
            error_x=dict(type="data", array=fi_sorted["std"].values, visible=True)
                   if "std" in fi_sorted.columns else None,
            marker_color="#2e7d32",
            text=fi_sorted["importance"].round(4),
            textposition="outside",
        ))
        fig_fi.update_layout(
            title=f"Top {top_n_fi} Features — Permutation Importance",
            xaxis_title="Importance (RMSE increase when shuffled)",
            yaxis=dict(categoryorder="total ascending"),
            plot_bgcolor="white",
            height=max(350, top_n_fi * 22),
        )
        fig_fi.update_xaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("📋 Toàn bộ bảng feature importance"):
            st.dataframe(fi.sort_values("importance", ascending=False), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Ablation Study
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Ablation Study — Proposed Model (Chronos + HGB Ensemble)")
    st.caption("Đánh giá đóng góp từng thành phần — Validation set, Protocol C (H=7, no oracle lags)")

    abl_df = load_ablation()
    if abl_df.empty:
        st.warning("Chưa có ablation_proposed_b1b6_summary.csv. Hãy chạy cell 56 trong notebook.")
    else:
        # ── Metric selector ───────────────────────────────────────────────────
        abl_metric = st.selectbox(
            "Metric", ["val_MAE", "val_RMSE", "val_sMAPE"],
            format_func=lambda x: x.replace("val_", ""),
            key="abl_met",
        )

        # Reference = FULL model row
        ref_row = abl_df[abl_df["ablation"] == "FULL"]
        ref_val = float(ref_row[abl_metric].values[0]) if not ref_row.empty else None

        # ── Bar chart ─────────────────────────────────────────────────────────
        fig_abl = go.Figure()
        for _, row in abl_df.iterrows():
            is_ref   = row["ablation"] == "FULL"
            val      = float(row[abl_metric])
            worse    = (ref_val is not None) and (val > ref_val) and not is_ref
            color    = "#2e7d32" if is_ref else ("#ef5350" if worse else "#66bb6a")
            delta_str = "" if is_ref or ref_val is None else f"  Δ{val - ref_val:+.2f}"
            fig_abl.add_trace(go.Bar(
                x=[row["label"]],
                y=[val],
                marker_color=color,
                text=[f"{val:.3f}{delta_str}"],
                textposition="outside",
                name=row["label"],
                showlegend=False,
                customdata=[[row["ablation"], row.get("alpha", ""), row.get("n_features", "")]],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{abl_metric.replace('val_', '')}: %{{y:.4f}}<br>"
                    "ID: %{customdata[0]}  α=%{customdata[1]}  feats=%{customdata[2]}"
                    "<extra></extra>"
                ),
            ))
        if ref_val is not None:
            fig_abl.add_hline(
                y=ref_val, line_dash="dash", line_color="#2e7d32", line_width=1.5,
                annotation_text=f"Full model = {ref_val:.3f}",
                annotation_position="top right",
                annotation_font_color="#2e7d32",
            )
        fig_abl.update_layout(
            title=f"Ablation — {abl_metric.replace('val_', '')}  (lower = better)",
            xaxis_tickangle=-20,
            yaxis_title=abl_metric.replace("val_", ""),
            plot_bgcolor="white",
            height=420,
            margin=dict(t=60, b=20),
        )
        fig_abl.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig_abl, use_container_width=True)

        # ── Delta table ───────────────────────────────────────────────────────
        st.subheader("Δ so với Full Proposed Model")
        disp = abl_df[["ablation", "label", "alpha", "n_features",
                        "val_MAE", "val_RMSE", "val_sMAPE"]].copy()
        for m in ["val_MAE", "val_RMSE", "val_sMAPE"]:
            ref_m = float(abl_df.loc[abl_df["ablation"] == "FULL", m].values[0]) \
                    if not abl_df[abl_df["ablation"] == "FULL"].empty else float("nan")
            disp[f"Δ{m.replace('val_','')}"] = (disp[m] - ref_m).round(3)
        disp = disp.rename(columns={
            "ablation": "ID", "label": "Ablation",
            "alpha": "α", "n_features": "#Feats",
        })
        st.dataframe(
            disp.style.background_gradient(
                subset=[f"Δ{m.replace('val_','')}" for m in ["val_MAE","val_RMSE","val_sMAPE"]],
                cmap="RdYlGn_r",
            ),
            use_container_width=True,
        )

        # ── Interpretation ────────────────────────────────────────────────────
        with st.expander("📖 Giải thích các ablation"):
            st.markdown("""
| ID | Ablation | Mục đích |
|---|---|---|
| **FULL** | Full Proposed (HPO) | Baseline reference |
| **B1** | Chronos Only (α=1) | Bỏ HGB — chỉ dùng foundation model |
| **B2** | HGB Only (α=0) | Bỏ Chronos — chỉ dùng tree-based model |
| **B3** | No HPO | Dùng default HGB params, bỏ Optuna optimization |
| **B4** | No Lag Features | Bỏ lag_1..lag_28 — kiểm tra đóng góp autocorrelation |
| **B5** | No Calendar | Bỏ dayofweek, month, dayofyear |
| **B6** | No Exogenous | Bỏ Demand, Inventory Level, Discount, Promotion, v.v. |

**Màu sắc:** 🟢 Tốt hơn / bằng Full model · 🔴 Kém hơn Full model
            """)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — About
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("About — Proposed Model")
    st.markdown("""
    ### Chronos-Bolt-Small + HGB Ensemble

    **Proposed model** kết hợp hai thành phần:

    | Thành phần | Mô tả |
    |---|---|
    | **Chronos-Bolt-Small** | Foundation model zero-shot của Amazon (pre-trained), dự báo không cần fine-tune |
    | **HistGradientBoostingRegressor** | Tree-based model với lag features, rolling stats, calendar features |
    | **Blend** | `ŷ = α × Chronos + (1-α) × HGB` với α được tối ưu bởi HPO |

    ### Splits

    | Period | Range | #Days |
    |---|---|---|
    | Train | 2022-01-01 → 2023-06-30 | ~547 |
    | Validation | 2023-07-01 → 2023-10-31 | ~123 |
    | Test | 2023-11-01 → 2024-01-30 | ~91 |

    ### Protocol C (no oracle lags)
    Mỗi series được dự báo **H=7 bước** từ forecast origin, không dùng giá trị thực tế tương lai làm feature.
    Lag features tại bước h=2..7 được thay bằng giá trị predicted từ bước trước (recursive forecasting).

    ### HPO
    Tối ưu đồng thời 6 HGB hyperparams + blend weight α bằng **Optuna**:
    - Strategy 1: **TPE** (Bayesian) — 50 trials
    - Strategy 2: **Random Search** — 100 trials

    ### Files
    """)

    result_files = list(RESULT_DIR.glob("*"))
    if result_files:
        file_df = pd.DataFrame([
            {"File": f.name, "Size": f"{f.stat().st_size/1024:.1f} KB"}
            for f in sorted(result_files) if f.is_file()
        ])
        st.dataframe(file_df, use_container_width=True)
