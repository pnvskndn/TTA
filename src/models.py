"""
models.py
Three models for the Technology Adoption Analyzer:

  Model 1: XGBoost + SHAP  — what factors explain adoption?
  Model 2: Prophet          — how fast will adoption grow?
  Model 3: K-Means          — which countries are similar?
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET_COL = "renewable_energy_pct"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — XGBoost + SHAP
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Trains XGBoost regressor and evaluates on test set.
    Returns trained model + evaluation metrics.
    """
    print("\n── Model 1: XGBoost ─────────────────────────────")

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae":  mean_absolute_error(y_test, y_pred),
        "r2":   r2_score(y_test, y_pred),
    }

    print(f"  RMSE: {metrics['rmse']:.2f}  |  "
          f"MAE: {metrics['mae']:.2f}  |  "
          f"R²: {metrics['r2']:.3f}")

    # Save model
    path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved to: {path}")

    return model, metrics, y_pred


def compute_shap_values(model, X_train, X_test, feature_names):
    """
    Computes SHAP values using TreeExplainer.
    Returns shap_values array and the explainer object.

    SHAP tells you: for each prediction, how much did each
    feature push the prediction up or down from the baseline?
    """
    print("\n── SHAP Analysis ────────────────────────────────")
    explainer = shap.TreeExplainer(model)

    # Compute on test set (faster than full dataset)
    shap_values = explainer.shap_values(X_test)

    # Global feature importance — mean |SHAP| per feature
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n  Global feature importance (mean |SHAP|):")
    for _, row in importance.head(8).iterrows():
        bar = "█" * int(row["mean_abs_shap"] / importance["mean_abs_shap"].max() * 20)
        print(f"  {row['feature']:<35} {bar} {row['mean_abs_shap']:.3f}")

    return shap_values, explainer, importance


def get_country_shap(model, X_test, df_test, country_code, feature_names):
    """
    Returns SHAP values for a specific country's most recent year.
    Used in the app to show "what drives adoption in Country X?"
    """
    country_mask = df_test["country_code"] == country_code
    if not country_mask.any():
        return None, None

    # Most recent year for this country
    idx = df_test[country_mask]["year"].idxmax()
    row = X_test.loc[[idx]]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row)[0]

    result = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals,
        "abs_shap": np.abs(shap_vals),
        "feature_value": row.values[0],
    }).sort_values("abs_shap", ascending=False)

    return result, explainer.expected_value


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Prophet time series forecasting
# ══════════════════════════════════════════════════════════════════════════════

def forecast_country(df_full, country_code, periods=5):
    """
    Fits a Prophet model to one country's historical adoption data
    and forecasts `periods` years into the future.

    Returns:
        historical  — DataFrame of actual values
        forecast    — Prophet forecast DataFrame
        cagr        — Compound Annual Growth Rate over historical period
    """
    from prophet import Prophet

    country_df = df_full[df_full["country_code"] == country_code].copy()
    country_df = country_df.sort_values("year")

    if len(country_df) < 5:
        return None, None, None

    # Prophet requires columns named 'ds' and 'y'
    prophet_df = country_df[["year", TARGET_COL]].rename(
        columns={"year": "ds", TARGET_COL: "y"}
    )
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")

    # Clip to reasonable range — renewable % can't be negative or > 100
    prophet_df["y"] = prophet_df["y"].clip(0, 100)

    # Fit model
    m = Prophet(
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,   # low = smoother trend
        interval_width=0.80,            # 80% confidence interval
        seasonality_mode="additive",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(prophet_df)

    # Forecast
    future = m.make_future_dataframe(periods=periods, freq="YS")
    forecast = m.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(0, 100)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(0, 100)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(0, 100)

    # CAGR calculation
    start_val = prophet_df["y"].iloc[0]
    end_val   = prophet_df["y"].iloc[-1]
    n_years   = len(prophet_df) - 1

    if start_val > 0 and n_years > 0:
        cagr = (end_val / start_val) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    historical = prophet_df.copy()

    return historical, forecast, cagr


def forecast_all_countries(df_full, countries, periods=5):
    """
    Runs Prophet for all countries. Returns a dict: country_code -> forecast.
    Takes ~30 seconds for 40 countries.
    """
    print(f"\n── Model 2: Prophet forecasting ({len(countries)} countries) ──")
    results = {}

    for i, cc in enumerate(countries):
        hist, forecast, cagr = forecast_country(df_full, cc, periods)
        if forecast is not None:
            results[cc] = {
                "historical": hist,
                "forecast": forecast,
                "cagr": cagr,
            }
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(countries)}...")

    print(f"  Done. Forecasts for {len(results)} countries.")

    path = os.path.join(MODELS_DIR, "prophet_forecasts.pkl")
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Saved to: {path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — K-Means clustering
# ══════════════════════════════════════════════════════════════════════════════

def train_clustering(df_full, n_clusters=5):
    """
    Clusters countries by their adoption profile.
    Uses most recent year's features per country.

    Returns:
        cluster_df  — DataFrame with country_code, cluster, PCA coords
        kmeans      — fitted KMeans model
        scaler      — fitted StandardScaler
        pca         — fitted PCA model
    """
    from sklearn.metrics import silhouette_score

    print(f"\n── Model 3: K-Means clustering (k={n_clusters}) ──")

    # Use most recent year per country
    latest = (
        df_full.sort_values("year")
        .groupby("country_code")
        .last()
        .reset_index()
    )

    cluster_features = [
        TARGET_COL,
        "gdp_per_capita",
        "urbanization_pct",
        "co2_per_capita",
        "renewable_energy_total_pct",
    ]
    cluster_features = [f for f in cluster_features if f in latest.columns]
    latest_clean = latest.dropna(subset=cluster_features)

    X = latest_clean[cluster_features].values

    # Normalize — essential before K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find best k using silhouette score
    best_k, best_score = n_clusters, -1
    print("  Silhouette scores:")
    for k in range(3, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"    k={k}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

    print(f"  Best k = {best_k} (silhouette = {best_score:.3f})")

    # Final model with best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    cluster_df = latest_clean[["country_code"]].copy()
    cluster_df["cluster"] = labels
    cluster_df["pc1"] = coords[:, 0]
    cluster_df["pc2"] = coords[:, 1]
    cluster_df[TARGET_COL] = latest_clean[TARGET_COL].values

    # Cluster profiles — mean feature values per cluster
    cluster_df_with_features = latest_clean.copy()
    cluster_df_with_features["cluster"] = labels
    profiles = cluster_df_with_features.groupby("cluster")[cluster_features].mean()

    print("\n  Cluster profiles (mean values):")
    print(profiles.round(1).to_string())

    # Save
    path = os.path.join(MODELS_DIR, "clustering.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "cluster_df": cluster_df,
            "kmeans": kmeans,
            "scaler": scaler,
            "pca": pca,
            "profiles": profiles,
            "features": cluster_features,
        }, f)
    print(f"\n  Saved to: {path}")

    return cluster_df, kmeans, scaler, pca, profiles


# ══════════════════════════════════════════════════════════════════════════════
# POLICY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_policy(model, base_features_row, feature_names,
                    gdp_change_pct=0, co2_price_effect=0,
                    urbanization_change=0):
    """
    What-if scenario analysis.
    Modifies input features and returns new prediction.

    Args:
        base_features_row: dict of feature_name -> current value
        gdp_change_pct: % change in GDP per capita (+20 = 20% increase)
        co2_price_effect: increase in CO2 per capita pressure (proxy for carbon pricing)
        urbanization_change: percentage point change in urbanization

    Returns:
        baseline_pred, modified_pred, change
    """
    base_df = pd.DataFrame([base_features_row], columns=feature_names)
    baseline_pred = float(model.predict(base_df)[0])

    modified = base_features_row.copy()

    # Apply changes
    if "log_gdp_per_capita" in modified:
        # Convert % GDP change to log scale
        gdp_multiplier = 1 + (gdp_change_pct / 100)
        modified["log_gdp_per_capita"] = (
            np.log1p(np.expm1(modified["log_gdp_per_capita"]) * gdp_multiplier)
        )

    if "co2_per_capita" in modified:
        modified["co2_per_capita"] = max(
            0, modified["co2_per_capita"] + co2_price_effect
        )

    if "urbanization_pct" in modified:
        modified["urbanization_pct"] = min(
            100, max(0, modified["urbanization_pct"] + urbanization_change)
        )

    mod_df = pd.DataFrame([modified], columns=feature_names)
    modified_pred = float(model.predict(mod_df)[0])
    modified_pred = max(0, min(100, modified_pred))

    change = modified_pred - baseline_pred

    return baseline_pred, modified_pred, change


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import fetch_world_bank_data
    from preprocessor import run_pipeline, temporal_train_test_split, build_feature_matrix

    print("Loading data...")
    raw = fetch_world_bank_data(use_cache=True)
    df  = run_pipeline(raw)

    train_df, test_df = temporal_train_test_split(df)
    X_train, y_train, features = build_feature_matrix(train_df)
    X_test,  y_test,  _        = build_feature_matrix(test_df)

    # Train all three models
    model, metrics, y_pred = train_xgboost(X_train, y_train, X_test, y_test)
    shap_vals, explainer, importance = compute_shap_values(
        model, X_train, X_test, features
    )

    forecasts = forecast_all_countries(df, df["country_code"].unique()[:10])  # test on 10

    cluster_df, kmeans, scaler, pca, profiles = train_clustering(df)
