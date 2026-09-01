"""
app.py — Technology Adoption Analyzer
Consulting-exhibit styling — forced light mode, navy ink, electric blue accent,
mono numerals for all stats. Data-forward.

Phase 1: structural benchmark model (no lag features) + per-country gap.
Phase 2: rolling-origin backtests -> measured historical accuracy in-app.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import fetch_world_bank_data, load_country_metadata
from preprocessor import run_pipeline, temporal_train_test_split, build_feature_matrix
from models import (
    train_xgboost, compute_shap_values,
    forecast_all_countries, train_clustering,
    get_country_shap, simulate_policy,
)
from visualizations import (
    plot_historical_trend, plot_forecast,
    plot_shap_waterfall, plot_global_feature_importance,
    plot_choropleth, plot_cluster_scatter,
    plot_policy_simulation, plot_region_comparison,
)
from structural import (
    train_structural_model, compute_structural_gaps, simulate_scenario,
)
from backtest import get_or_run_backtests, country_accuracy

st.set_page_config(
    page_title="Technology Adoption Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

TARGET_COL = "renewable_energy_pct"
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
BACKTEST_CACHE = os.path.join(OUTPUTS_DIR, "backtest_results.pkl")


# ── CSS ────────────────────────────────────────────────────────────────────────
# Design tokens (mirrored in src/visualizations.py and .streamlit/config.toml):
#   INK    #051C2C   deep navy — text, headlines, primary data
#   ACCENT #2251FF   electric blue — emphasis, interaction, selected states
#   SLATE  #5B6B7C   secondary text and labels
#   BORDER #E3E8ED   hairlines
#   PAPER  #F5F7F9   panel fills
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* ── Force light mode ── */
    :root { color-scheme: light only !important; }

    .stApp,
    .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    .main,
    .block-container {
        background-color: #FFFFFF !important;
        color: #051C2C !important;
    }

    /* ── Report band — signature accent across the very top ── */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #051C2C 0%, #2251FF 100%);
        z-index: 9999;
    }

    /* ── Global typography ── */
    html, body, [class*="css"], p, span, div, li {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: #051C2C;
    }

    /* ── Restore icon fonts — Streamlit icons are Material Symbols ligatures.
       Without this, icon spans render as raw text like "keyboard_double_arrow_left" ── */
    [data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"] *,
    span[class*="material-symbols"],
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    }

    /* ── Hide chrome ── */
    #MainMenu, footer, header, .stDeployButton,
    [data-testid="stToolbar"] { display: none !important; }

    /* ── Main container ── */
    .main .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem !important;
        max-width: 1280px !important;
    }

    /* ── Headings ── */
    h1 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: 2.05rem !important;
        font-weight: 600 !important;
        color: #051C2C !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
        margin-bottom: 0.2rem !important;
    }
    h2 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #051C2C !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        margin: 1.5rem 0 0.6rem 0 !important;
    }
    h3 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        color: #5B6B7C !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }

    /* ── Caption ── */
    .stCaption p,
    [data-testid="stCaptionContainer"] p {
        font-size: 0.78rem !important;
        color: #5B6B7C !important;
        font-style: normal !important;
        line-height: 1.55 !important;
    }

    /* ── Metric cards — exhibit boxes ── */
    [data-testid="stMetric"],
    [data-testid="metric-container"] {
        background: #FFFFFF !important;
        border: 1px solid #E3E8ED !important;
        border-top: 3px solid #051C2C !important;
        border-radius: 2px !important;
        padding: 0.75rem 0.9rem !important;
        box-shadow: 0 1px 3px rgba(5, 28, 44, 0.06) !important;
    }
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] > div,
    [data-testid="metric-container"] > label > div {
        font-size: 0.66rem !important;
        font-weight: 600 !important;
        color: #5B6B7C !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        white-space: normal !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.45rem !important;
        font-weight: 500 !important;
        color: #051C2C !important;
        line-height: 1.25 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] > div {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        white-space: normal !important;
        line-height: 1.35 !important;
    }
    /* Sidebar metrics live in half-width columns — size down to fit */
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"] > div {
        font-size: 1.15rem !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #F5F7F9 !important;
        border-right: 1px solid #E3E8ED !important;
    }
    [data-testid="stSidebar"] p { font-size: 0.83rem !important; }

    .sidebar-title {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #051C2C !important;
        letter-spacing: -0.01em !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #051C2C !important;
        margin-bottom: 1rem !important;
    }

    /* ── Radio ── */
    [data-testid="stRadio"] > label { display: none !important; }
    [data-testid="stRadio"] label[data-baseweb="radio"] {
        padding: 0.35rem 0 !important;
    }
    [data-testid="stRadio"] p {
        font-size: 0.84rem !important;
        color: #051C2C !important;
    }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > label > div {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: #5B6B7C !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-color: #E3E8ED !important;
        border-radius: 2px !important;
    }
    [data-baseweb="select"] * {
        font-size: 0.84rem !important;
        background-color: #FFFFFF !important;
        color: #051C2C !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] > label > div {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: #5B6B7C !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stSlider"] { padding: 0.2rem 0 !important; }

    /* ── Tabs ── */
    [data-baseweb="tab-list"] {
        gap: 0 !important;
        background: transparent !important;
        border-bottom: 1px solid #E3E8ED !important;
    }
    [data-baseweb="tab"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #5B6B7C !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        padding: 0.55rem 1.1rem !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        color: #051C2C !important;
        border-bottom: 2px solid #2251FF !important;
    }
    [data-testid="stTabsContent"] {
        background: #FFFFFF !important;
        padding-top: 0.75rem !important;
    }

    /* ── Button ── */
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        background: #051C2C !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 0.45rem 1.4rem !important;
        transition: background 0.15s ease !important;
    }
    .stButton > button:hover { background: #2251FF !important; }

    /* ── Info box ── */
    [data-testid="stAlert"],
    [data-testid="stInfo"] {
        background: #F5F7F9 !important;
        border: 1px solid #E3E8ED !important;
        border-left: 3px solid #2251FF !important;
        border-radius: 0 2px 2px 0 !important;
        color: #051C2C !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stInfo"] p { font-size: 0.83rem !important; }
    [data-testid="stAlert"] svg,
    [data-testid="stInfo"] svg { display: none !important; }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid #E3E8ED !important;
        margin: 1.25rem 0 !important;
    }

    /* ── Static table (Top 10) — real HTML, fully stylable ── */
    [data-testid="stTable"] {
        background: #FFFFFF !important;
    }
    [data-testid="stTable"] table {
        border-collapse: collapse !important;
        border: 1px solid #E3E8ED !important;
        border-top: 3px solid #051C2C !important;
        width: 100% !important;
    }
    [data-testid="stTable"] th {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.66rem !important;
        font-weight: 600 !important;
        color: #5B6B7C !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        background: #F5F7F9 !important;
        border-bottom: 1px solid #E3E8ED !important;
        padding: 0.5rem 0.8rem !important;
        text-align: left !important;
    }
    [data-testid="stTable"] td {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        color: #051C2C !important;
        border-bottom: 1px solid #EEF1F4 !important;
        padding: 0.45rem 0.8rem !important;
    }
    [data-testid="stTable"] tr:last-child td {
        border-bottom: none !important;
    }

    /* ── Dropdown menus ── */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    [role="option"] {
        background: #FFFFFF !important;
        color: #051C2C !important;
    }

    /* ── Custom components ── */
    .eyebrow {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.66rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #5B6B7C;
        margin-bottom: 0.5rem;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .eyebrow::before {
        content: "";
        width: 8px;
        height: 8px;
        background: #2251FF;
        flex: 0 0 auto;
    }
    .page-header {
        border-bottom: 2px solid #051C2C;
        padding-bottom: 0.7rem;
        margin-bottom: 1.4rem;
    }
    .page-subtitle {
        font-size: 0.8rem;
        color: #5B6B7C;
        margin: 0.15rem 0 0 0;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .section-rule {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.68rem;
        font-weight: 600;
        color: #5B6B7C;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-top: 1px solid #E3E8ED;
        padding-top: 0.7rem;
        margin: 1.1rem 0 0.65rem 0;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .section-rule::before {
        content: "";
        width: 6px;
        height: 6px;
        background: #2251FF;
        flex: 0 0 auto;
    }
    .stat-label {
        font-size: 0.63rem;
        font-weight: 600;
        color: #5B6B7C;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.3rem;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title, subtitle="", eyebrow=""):
    eb = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    sub = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="page-header">{eb}<h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def section_label(text):
    st.markdown(
        f'<div class="section-rule">{text}</div>',
        unsafe_allow_html=True,
    )


def scope_line(df):
    """Dataset provenance line shown as the page eyebrow."""
    n = df["country_code"].nunique()
    y0, y1 = int(df["year"].min()), int(df["year"].max())
    return f"Renewable energy · {n} countries · {y0}–{y1}"


# ── Data + models ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    raw = fetch_world_bank_data(use_cache=True)
    return run_pipeline(raw)


@st.cache_resource(show_spinner="Training models — first run also computes backtests and takes a few minutes...")
def load_models(_df):
    train_df, test_df = temporal_train_test_split(_df)
    X_train, y_train, features = build_feature_matrix(train_df)
    X_test, y_test, _ = build_feature_matrix(test_df)

    model, metrics, _ = train_xgboost(X_train, y_train, X_test, y_test)
    shap_vals, explainer, importance = compute_shap_values(model, X_train, X_test, features)

    countries = _df["country_code"].dropna().unique().tolist()
    forecasts = forecast_all_countries(_df, countries)
    cluster_df, kmeans, scaler, pca, profiles = train_clustering(_df)

    # Phase 1 — structural benchmark (no lag features) + per-country gap
    s_model, s_feats, s_metrics = train_structural_model(_df)
    gaps = compute_structural_gaps(_df, s_model, s_feats)

    # Phase 2 — rolling-origin backtests (cached to disk after first run)
    backtests = get_or_run_backtests(_df, BACKTEST_CACHE)

    return {
        "model": model, "metrics": metrics, "features": features,
        "shap_values": shap_vals, "explainer": explainer,
        "importance": importance, "forecasts": forecasts,
        "cluster_df": cluster_df, "test_df": test_df, "X_test": X_test,
        "structural_model": s_model, "structural_features": s_feats,
        "structural_metrics": s_metrics, "gaps": gaps,
        "backtests": backtests,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar(df, models):
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">Technology Adoption Analyzer</div>',
            unsafe_allow_html=True,
        )
        st.caption(scope_line(df))
        st.markdown("<br>", unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["Country Explorer", "Factor Analysis", "World Map", "Structural Benchmark"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown('<div class="stat-label">Model performance</div>', unsafe_allow_html=True)
        m = models["metrics"]
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{m['rmse']:.2f}")
        c2.metric("R²", f"{m['r2']:.3f}")
        st.caption("XGBoost · Temporal split\nTrain 2001–2018 · Test 2019–2021")

        st.markdown("---")
        st.caption("Contributing to UN SDG 7 — Affordable and Clean Energy")

    return page


# ── Page 1 ─────────────────────────────────────────────────────────────────────
def page_country_explorer(df, models):
    page_header(
        "Country Explorer",
        "Historical trend, 5-year forecast, and factor attribution per country",
        eyebrow=scope_line(df),
    )

    meta = load_country_metadata()
    opts = {
        r["country_name"]: r["country_code"]
        for _, r in meta.iterrows()
        if r["country_code"] in df["country_code"].values
    }

    col_l, col_r = st.columns([1, 3])

    with col_l:
        name = st.selectbox(
            "Country",
            sorted(opts.keys()),
            index=sorted(opts.keys()).index("India"),
        )
        code = opts[name]
        st.markdown("<br>", unsafe_allow_html=True)

        cdf = df[df["country_code"] == code].sort_values("year")
        if not cdf.empty:
            lat, ear = cdf.iloc[-1], cdf.iloc[0]
            if pd.notna(lat[TARGET_COL]) and pd.notna(ear[TARGET_COL]):
                st.metric(
                    f"Current ({int(lat['year'])})",
                    f"{lat[TARGET_COL]:.1f}%",
                    delta=f"{lat[TARGET_COL]-ear[TARGET_COL]:+.1f}pp since {int(ear['year'])}",
                )
            if code in models["forecasts"]:
                cagr = models["forecasts"][code]["cagr"]
                st.metric("CAGR", f"{cagr*100:+.2f}%/yr")

        # Phase 1 — structural gap card
        g = models["gaps"]
        grow = g[g["country_code"] == code]
        if len(grow):
            st.metric(
                "Structural gap",
                f"{grow['structural_gap_pp'].values[0]:+.1f} pp",
                help=(
                    "Actual adoption minus the level this country's structural "
                    "fundamentals (GDP, urbanization, education, coal share, "
                    "consumption) typically support. Positive = over-performer; "
                    "negative = headroom. Associational benchmark, not causal."
                ),
            )

        section_label("Similar countries")
        cdf2 = models["cluster_df"]
        cl = cdf2[cdf2["country_code"] == code]["cluster"].values
        if len(cl) > 0:
            peers = cdf2[
                (cdf2["cluster"] == cl[0]) & (cdf2["country_code"] != code)
            ]["country_code"].tolist()
            for cc in peers[:5]:
                row = meta[meta["country_code"] == cc]
                if len(row) > 0:
                    st.caption(f"— {row['country_name'].values[0]}")

    with col_r:
        t1, t2, t3 = st.tabs(["Historical trend", "5-year forecast", "Factor attribution"])

        with t1:
            st.plotly_chart(plot_historical_trend(df, code, name), use_container_width=True)

        with t2:
            if code in models["forecasts"]:
                fc = models["forecasts"][code]
                st.plotly_chart(plot_forecast(fc["historical"], fc["forecast"], name), use_container_width=True)
                # Phase 2 — measured historical accuracy
                acc = country_accuracy(models["backtests"], code)
                if acc:
                    st.caption(
                        f"Backtested accuracy — rolling-origin forecasts for {name} have "
                        f"historically missed by ±{acc['mae']:.1f} pp "
                        f"(1–{acc['max_horizon']} yr horizons, n={acc['n']}); "
                        f"80% intervals covered {acc['coverage']:.0%} of outcomes."
                    )
            else:
                st.info("Insufficient historical data for this country.")

        with t3:
            sr, bv = get_country_shap(models["model"], models["X_test"], models["test_df"], code, models["features"])
            if sr is not None:
                st.plotly_chart(plot_shap_waterfall(sr, name, bv), use_container_width=True)
                st.caption("SHAP values show each factor's contribution to the model prediction. Blue = increases adoption. Red = decreases.")
            else:
                st.info("SHAP attribution not available for this country in the test set.")


# ── Page 2 ─────────────────────────────────────────────────────────────────────
def page_factor_analysis(df, models):
    page_header(
        "Factor Analysis",
        "Which socioeconomic and policy factors most strongly predict renewable energy adoption?",
        eyebrow=scope_line(df),
    )

    # Headline stats — what the model actually found
    imp = models["importance"]
    top_feat = str(imp.iloc[0]["feature"])
    top_val = float(imp.iloc[0]["mean_abs_shap"])
    share = 100 * top_val / float(imp["mean_abs_shap"].sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Top driver", top_feat.replace("_", " "))
    k2.metric("Mean |SHAP| — top driver", f"{top_val:.2f}")
    k3.metric("Share of total importance", f"{share:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section_label("Global feature importance")
        st.plotly_chart(plot_global_feature_importance(models["importance"]), use_container_width=True)
        st.caption("Mean absolute SHAP value across test-set predictions (2019–2021).")

    with c2:
        section_label("Distribution by region")
        yr = st.selectbox("Year", sorted(df["year"].unique(), reverse=True), index=0)
        st.plotly_chart(plot_region_comparison(df, year=yr), use_container_width=True)

    st.markdown("---")
    section_label("Country clusters")
    k = int(models["cluster_df"]["cluster"].nunique())
    st.caption(f"K-Means (k={k}) on adoption profile and economic features. PCA to 2D. Point size ∝ renewable adoption.")
    st.plotly_chart(plot_cluster_scatter(models["cluster_df"]), use_container_width=True)


# ── Page 3 ─────────────────────────────────────────────────────────────────────
def page_world_map(df, models):
    page_header(
        "World Map",
        "Renewable energy adoption by country — drag the slider to animate over time",
        eyebrow=scope_line(df),
    )

    yr = st.select_slider(
        "Year", options=sorted(df["year"].unique()),
        value=2020, label_visibility="collapsed",
    )

    nc = "country_name_x" if "country_name_x" in df.columns else "country_name"

    # Headline stats for the selected year
    ydf = df[df["year"] == yr].dropna(subset=[TARGET_COL, nc])
    if not ydf.empty:
        leader = ydf.loc[ydf[TARGET_COL].idxmax()]
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Global average — {yr}", f"{ydf[TARGET_COL].mean():.1f}%")
        k2.metric("Leader", str(leader[nc]), delta=f"{leader[TARGET_COL]:.1f}%", delta_color="off")
        k3.metric("Countries above 50%", f"{int((ydf[TARGET_COL] >= 50).sum())} of {len(ydf)}")

    st.plotly_chart(plot_choropleth(df, year=yr), use_container_width=True)

    st.markdown("---")
    section_label(f"Top 10 — {yr}")

    top10 = (
        df[df["year"] == yr][[nc, TARGET_COL, "region"]]
        .dropna()
        .sort_values(TARGET_COL, ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10.index += 1
    top10.columns = ["Country", "Renewable %", "Region"]
    top10["Renewable %"] = top10["Renewable %"].round(1).astype(str) + "%"
    st.table(top10)


# ── Page 4 — Structural Benchmark (Phase 1) ────────────────────────────────────
def page_structural_benchmark(df, models):
    page_header(
        "Structural Benchmark",
        "What adoption level do a country's fundamentals typically support — and where does it actually sit?",
        eyebrow=scope_line(df),
    )

    meta = load_country_metadata()
    opts = {
        r["country_name"]: r["country_code"]
        for _, r in meta.iterrows()
        if r["country_code"] in df["country_code"].values
    }

    s_model = models["structural_model"]
    s_feats = models["structural_features"]
    s_metrics = models["structural_metrics"]
    gaps = models["gaps"]

    cl, cr = st.columns([1, 2])

    with cl:
        name = st.selectbox(
            "Country",
            sorted(opts.keys()),
            index=sorted(opts.keys()).index("India"),
            key="sb_country",
        )
        code = opts[name]

        section_label("Scenario levers")
        gdp = st.slider("GDP per capita change (%)", -20, 100, 0, 5, help="Economic growth/contraction")
        urb = st.slider("Urbanization change (pp)", -10, 20, 0, 1, help="Change in urban population share")
        coal = st.slider("Coal electricity change (pp)", -20, 20, 0, 1, help="Change in coal's share of electricity")

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Run scenario")

        if s_metrics:
            st.caption(
                f"Structural model test R²: {s_metrics['r2']:.2f} — intentionally far "
                f"below the forecasting model's. Without lag features it measures only "
                f"what fundamentals explain; the remainder is history, endowment, and policy."
            )

    with cr:
        # Benchmark position — always visible
        grow = gaps[gaps["country_code"] == code]
        cdf = df[df["country_code"] == code].sort_values("year").dropna(subset=s_feats + [TARGET_COL])

        if grow.empty or cdf.empty:
            st.info("Insufficient data for this country.")
            return

        r = grow.iloc[0]
        b1, b2, b3 = st.columns(3)
        b1.metric(f"Actual ({int(r['year'])})", f"{r[TARGET_COL]:.1f}%")
        b2.metric("Structurally expected", f"{r['expected_pct']:.1f}%")
        b3.metric("Structural gap", f"{r['structural_gap_pp']:+.1f} pp")

        if run:
            base_row = cdf.iloc[-1]
            baseline, modified, change = simulate_scenario(
                s_model, s_feats, base_row,
                gdp_change_pct=gdp, urbanization_change=urb, coal_change=coal,
            )
            fig = plot_policy_simulation(baseline, modified, name, scenario_label="Scenario")
            fig.update_layout(title_text=f"Structural Benchmark — {name}")
            st.plotly_chart(fig, use_container_width=True)

            direction = "rises" if change >= 0 else "falls"
            st.info(
                f"Countries with **{name}**'s current fundamentals typically sit near "
                f"**{baseline:.1f}%**. Under this scenario the benchmark {direction} to "
                f"**{modified:.1f}%** ({change:+.1f} pp)."
            )
            st.caption(
                "This is an associational benchmark from a model trained without "
                "path-dependency terms — it shows what fundamentals correlate with, "
                "not what a policy would cause. Actual transitions also depend on "
                "energy endowments, existing infrastructure, and policy execution."
            )
        else:
            section_label(f"{name} — actual vs structural expectation")
            st.plotly_chart(plot_historical_trend(df, code, name), use_container_width=True)
            st.caption(
                "Positive gap = over-performing its fundamentals (endowment/policy at work). "
                "Negative gap = structural headroom that history has not yet delivered."
            )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    with st.spinner("Loading..."):
        df = load_data()
        models = load_models(df)

    page = sidebar(df, models)

    if page == "Country Explorer":
        page_country_explorer(df, models)
    elif page == "Factor Analysis":
        page_factor_analysis(df, models)
    elif page == "World Map":
        page_world_map(df, models)
    elif page == "Structural Benchmark":
        page_structural_benchmark(df, models)


if __name__ == "__main__":
    main()