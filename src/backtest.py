"""
backtest.py — Phase 2: Rolling-origin backtesting for per-country forecasts.

Folds (train-cutoff -> predicted years):
    <=2013 -> 2014-2016
    <=2016 -> 2017-2019
    <=2019 -> 2020-2021

For every backtested prediction we record the absolute error and whether the
actual fell inside the 80% interval. Aggregated per country, this replaces
promised precision with measured accuracy: "forecasts for X have historically
missed by +/-N pp; intervals covered M% of outcomes."

Results are cached to outputs/backtest_results.pkl. Everything fails
gracefully — if Prophet is unavailable or a country has too little history,
the app simply shows no accuracy line for it.
"""

import os
import pickle
import logging
import datetime

import numpy as np
import pandas as pd

TARGET_COL = "renewable_energy_pct"
FOLDS = [
    (2013, [2014, 2015, 2016]),
    (2016, [2017, 2018, 2019]),
    (2019, [2020, 2021]),
]
MIN_TRAIN_POINTS = 8

for _name in ("prophet", "cmdstanpy"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def _fit_predict(train_df: pd.DataFrame, horizon_years) -> pd.DataFrame:
    """Fit Prophet on annual data up to the cutoff; predict the horizon years."""
    from prophet import Prophet

    m = Prophet(
        interval_width=0.80,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    m.fit(train_df)
    future = pd.DataFrame(
        {"ds": pd.to_datetime([f"{y}-01-01" for y in horizon_years])}
    )
    fc = m.predict(future)
    fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    fc["year"] = fc["ds"].dt.year
    return fc


def run_backtest(df: pd.DataFrame, countries=None) -> dict:
    """Rolling-origin evaluation across all countries. Returns results dict."""
    if countries is None:
        countries = sorted(df["country_code"].dropna().unique())

    rows = []
    for cc in countries:
        series = (
            df[df["country_code"] == cc][["year", TARGET_COL]]
            .dropna()
            .drop_duplicates(subset="year")
            .sort_values("year")
        )
        if series.empty:
            continue

        for cutoff, horizon in FOLDS:
            train = series[series["year"] <= cutoff]
            if len(train) < MIN_TRAIN_POINTS:
                continue
            actuals = series[series["year"].isin(horizon)]
            if actuals.empty:
                continue

            train_df = pd.DataFrame({
                "ds": pd.to_datetime(train["year"].astype(int).astype(str) + "-01-01"),
                "y": train[TARGET_COL].to_numpy(),
            })
            try:
                fc = _fit_predict(train_df, horizon)
            except Exception as e:
                print(f"  backtest skipped {cc} @ cutoff {cutoff}: {e}")
                continue

            merged = actuals.merge(fc, on="year", how="inner")
            for _, r in merged.iterrows():
                rows.append({
                    "country_code": cc,
                    "cutoff": cutoff,
                    "year": int(r["year"]),
                    "horizon": int(r["year"] - cutoff),
                    "actual": float(r[TARGET_COL]),
                    "yhat": float(r["yhat"]),
                    "lo": float(r["yhat_lower"]),
                    "hi": float(r["yhat_upper"]),
                })

    per_prediction = pd.DataFrame(rows)
    if per_prediction.empty:
        return {"per_prediction": per_prediction, "per_country": pd.DataFrame(),
                "generated": datetime.datetime.now().isoformat()}

    per_prediction["abs_err"] = (per_prediction["actual"] - per_prediction["yhat"]).abs()
    per_prediction["covered"] = (
        (per_prediction["actual"] >= per_prediction["lo"])
        & (per_prediction["actual"] <= per_prediction["hi"])
    )

    per_country = (
        per_prediction.groupby("country_code")
        .agg(
            n_predictions=("abs_err", "size"),
            mae=("abs_err", "mean"),
            coverage=("covered", "mean"),
            max_horizon=("horizon", "max"),
        )
        .reset_index()
    )

    return {
        "per_prediction": per_prediction,
        "per_country": per_country,
        "generated": datetime.datetime.now().isoformat(),
    }


def get_or_run_backtests(df: pd.DataFrame, cache_path: str):
    """Load cached results if valid; otherwise run and cache. None on failure."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                results = pickle.load(f)
            if isinstance(results, dict) and "per_country" in results:
                return results
            print("Backtest cache malformed — recomputing.")
        except Exception as e:
            print(f"Backtest cache unreadable ({e}) — recomputing.")

    try:
        print("Running rolling-origin backtests (one-time, a few minutes)...")
        results = run_backtest(df)
    except Exception as e:
        print(f"Backtesting unavailable: {e}")
        return None

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
    except Exception as e:
        print(f"Could not cache backtest results: {e}")
    return results


def country_accuracy(results, country_code: str):
    """Per-country accuracy summary for display, or None if unavailable."""
    if not results or not isinstance(results, dict):
        return None
    pc = results.get("per_country")
    if pc is None or len(pc) == 0:
        return None
    row = pc[pc["country_code"] == country_code]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "mae": float(r["mae"]),
        "coverage": float(r["coverage"]),
        "n": int(r["n_predictions"]),
        "max_horizon": int(r["max_horizon"]),
    }