"""
structural.py — Phase 1: Structural Benchmark Model

A second XGBoost trained WITHOUT lag features. It predicts the adoption level
a country's structural fundamentals would typically support; the residual —
actual minus expected — is the structural gap, an empirical handle on
adoption barriers. All outputs are associational benchmarks, never causal
claims. Its test R² will be far below the lag model's: that is the point.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

TARGET_COL = "renewable_energy_pct"
TRAIN_END_YEAR = 2018   # mirrors the main pipeline's temporal split
TEST_END_YEAR = 2021

# Structural fundamentals only. Deliberately excluded: all lag features and
# renewable_energy_total_pct (outcome-adjacent — including them would defeat
# the benchmark's purpose). Open methodological question for the paper:
# coal share is real incumbent lock-in but also partly mechanical
# (shares of the same pie); removing it from this list is a one-line change.
STRUCTURAL_FEATURES = [
    "gdp_per_capita",
    "gdp_growth_rate",
    "urbanization_pct",
    "tertiary_education_pct",
    "electricity_consumption_pc",
    "coal_electricity_pct",
]


def train_structural_model(df: pd.DataFrame):
    """Train on 2001-2018, evaluate on 2019-2021. Returns (model, features, metrics)."""
    feats = [f for f in STRUCTURAL_FEATURES if f in df.columns]
    d = df.dropna(subset=[TARGET_COL] + feats).copy()

    train = d[d["year"] <= TRAIN_END_YEAR]
    test = d[(d["year"] > TRAIN_END_YEAR) & (d["year"] <= TEST_END_YEAR)]

    model = XGBRegressor(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(train[feats], train[TARGET_COL])

    metrics = {}
    if len(test):
        pred = np.clip(model.predict(test[feats]), 0, 100)
        err = test[TARGET_COL].to_numpy() - pred
        metrics = {
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(mean_absolute_error(test[TARGET_COL], pred)),
            "r2": float(r2_score(test[TARGET_COL], pred)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        }
    return model, feats, metrics


def compute_structural_gaps(df: pd.DataFrame, model, feats) -> pd.DataFrame:
    """
    Most recent year per country:
    structural_gap_pp = actual − structurally expected adoption.
    Positive = over-performer (e.g. Norway); negative = headroom (e.g. Gulf).
    """
    d = df.dropna(subset=[TARGET_COL] + feats).sort_values("year")
    latest = d.groupby("country_code").tail(1).copy()

    latest["expected_pct"] = np.clip(model.predict(latest[feats]), 0, 100)
    latest["structural_gap_pp"] = latest[TARGET_COL] - latest["expected_pct"]

    return (
        latest[["country_code", "year", TARGET_COL, "expected_pct", "structural_gap_pp"]]
        .reset_index(drop=True)
    )


def simulate_scenario(model, feats, base_row,
                      gdp_change_pct=0.0, urbanization_change=0.0,
                      coal_change=0.0):
    """
    Scenario on the structural benchmark: adjust fundamentals, re-predict.
    Returns (baseline, modified, change). Framing: 'countries with this
    profile typically sit at X%' — a benchmark shift, not a forecast.
    """
    base = {f: float(base_row[f]) for f in feats}
    baseline = float(np.clip(model.predict(pd.DataFrame([base])[feats])[0], 0, 100))

    mod = dict(base)
    if "gdp_per_capita" in mod:
        mod["gdp_per_capita"] *= (1 + gdp_change_pct / 100.0)
    if "urbanization_pct" in mod:
        mod["urbanization_pct"] = float(np.clip(mod["urbanization_pct"] + urbanization_change, 0, 100))
    if "coal_electricity_pct" in mod:
        mod["coal_electricity_pct"] = float(np.clip(mod["coal_electricity_pct"] + coal_change, 0, 100))

    modified = float(np.clip(model.predict(pd.DataFrame([mod])[feats])[0], 0, 100))
    return baseline, modified, modified - baseline