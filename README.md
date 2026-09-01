# Technology Adoption Analyzer

**Live app: <PASTE_STREAMLIT_URL_HERE>**

An ML research platform investigating what actually drives renewable energy
adoption across **44 countries (2001–2021)**, built on World Bank open data and
delivered as an interactive four-page dashboard.

---

## Research question

> What socioeconomic and policy factors most strongly predict renewable energy
> technology adoption — and how do these relationships vary across regions?

Connected to **UN SDG 7** (Affordable and Clean Energy) and **SDG 13**
(Climate Action). Being extended into a working paper.

## Key findings

**1. Path dependency dominates.** SHAP decomposition shows prior-year adoption
outweighs every economic and structural factor by a wide margin — consistent
with carbon lock-in (Unruh, 2000). Adoption behaves as momentum: policy must
overcome inertia, not merely create incentives.

**2. Wealth alone does not predict adoption.** Clustering separates high-income
renewable leaders (Norway, Denmark) from high-income fossil-dependent economies
(United States, Australia, the Gulf states) at comparable income levels.
Historical energy mix and policy stance are the differentiators.

**3. The structural gap quantifies headroom.** A second model trained *without*
path-dependency terms predicts the adoption level a country's fundamentals
typically support. The residual — actual minus expected — is an empirical
handle on barriers to adoption: positive for over-performers, sharply negative
where structural potential has not yet been realised.

Exact values are computed live — see the app rather than any static figure.

## The app

| Page | What it shows |
|---|---|
| Country Explorer | Historical trend, backtested 5-year forecast, SHAP attribution, structural gap, cluster peers |
| Factor Analysis | Global feature importance, regional distributions, country archetypes |
| World Map | Choropleth with year slider, headline stats, top-10 ranking |
| Structural Benchmark | Actual vs structurally expected adoption; scenario levers on fundamentals |

## Methodology

**Data.** Eight World Bank indicators, 44 countries, fetched via `wbdata` with a
direct REST API fallback. Target: renewable share of electricity generation
(`EG.ELC.RNEW.ZS`).

**Preprocessing.** Country-name normalization, forward/backward-fill with median
imputation, lag features, and a strict **temporal split** (train 2001–2018, test
2019–2021) — never random, to prevent leakage across time.

**Models.**
- **XGBoost + SHAP** — global and per-country attribution.
- **Prophet** — per-country 5-year forecasts with 80% intervals.
- **K-Means + PCA** — adoption archetypes, k by silhouette score.
- **Structural benchmark** — XGBoost on fundamentals only (GDP per capita and
  growth, urbanization, tertiary enrollment, coal share, consumption per
  capita), no lag terms. Its residual is the structural gap.

**Backtesting.** Rolling-origin evaluation (train→2013 predict 2014–16;
train→2016 predict 2017–19; train→2019 predict 2020–21) produces per-country
historical forecast error and 80%-interval coverage, displayed in-app. The
platform reports measured accuracy rather than claimed precision.

## Limitations

Attribution is **associational, not causal**. The forecasting model's R² is
near-ceiling because of autoregressive lag features — the contribution is the
SHAP decomposition and the structural gap, not raw predictive accuracy. The
structural model's R² is deliberately much lower: it measures only what
fundamentals explain. Annual data gives ~21 observations per country, and World
Bank indicators lag roughly two years.

## Pipeline resilience

The data layer fails loudly rather than corrupting silently: per-indicator
retries with backoff, a direct World Bank REST API fallback, refusal to cache
incomplete downloads, automatic detection and refetch of poisoned caches, World
Bank → display name normalization (official names like "Korea, Rep." and
"Viet Nam" otherwise fail to join with metadata), and hard alignment guards on
country metadata. One command traces country coverage through every stage:

```bash
TTA/bin/python3 diagnose_countries.py
```

## Run locally

```bash
git clone https://github.com/pnvskndn/TAA.git
cd TAA
python3 -m venv TTA
TTA/bin/python3 -m pip install -r requirements.txt
TTA/bin/python3 -m streamlit run app.py
```

First run takes several minutes (downloads data, trains models, computes
backtests); subsequent runs load from cache. The `numpy<2.0` and `xgboost<3.0`
pins in `requirements.txt` are load-bearing — numpy 2 breaks pickle
compatibility of cached artifacts, and xgboost 3 changed model serialization in
a way SHAP's tree parser cannot read.

## Project structure

```
├── app.py                    # 4-page Streamlit dashboard
├── diagnose_countries.py     # pipeline integrity check
├── requirements.txt
├── .streamlit/config.toml
├── src/
│   ├── data_loader.py        # WB API + REST fallback + metadata
│   ├── preprocessor.py       # cleaning, imputation, lags, temporal split
│   ├── models.py             # XGBoost/SHAP, Prophet, K-Means
│   ├── structural.py         # structural benchmark + gap
│   ├── backtest.py           # rolling-origin evaluation
│   └── visualizations.py     # Plotly exhibit library
├── data/                     # cache + processed (generated)
└── outputs/                  # trained models, backtest results
```

## Roadmap

- Working paper on adoption barriers (SSRN → technology-and-policy venue)
- Wider country panel for sharper structural benchmarks
- Grounded AI country briefs, recent-data integration, investor scorecard

---

Built by **Pranauv Skandhan** — physics undergraduate at SRM IST, Chennai,
pivoting to AI and data science. Part of a three-project research portfolio.
