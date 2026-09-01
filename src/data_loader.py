import os 
import json 
import time
import pickle
import datetime
import pandas as pd
import numpy as np

try:
  import wbdata
except ImportError:
  raise ImportError("Run: pip3 install wbdata")


TARGET_INDICATORS = {
  "EG.ELC.RNEW.ZS": "renewable_energy_pct", # % of total electricity form renewable sources
}

# FEATURE VARIABLES - explains adoption 
FEATURE_INDICATORS = {
  "NY.GDP.PCAP.CD": "gdp_per_capita", # GDP per capita in USD (wealth proxy)
  "SP.URB.TOTL.IN.ZS": "urbanization_pct", # urban population %
  "SE.TER.ENRR": "tertiary_education_pct", # tertiary school enrollment % (education proxy)
  "EG.ELC.COAL.ZS": "coal_electricity_pct", # % of electricity from coal (coal dependency)
  "EG.USE.ELEC.KH.PC": "electricity_consumption_pc", # electric power consumption per capita
  "EG.FEC.RNEW.ZS": "renewable_energy_total_pct", # renewable energy % of total energy (broader energy mix)
  "NY.GDP.PCAP.KD.ZG": "gdp_growth_rate",        # GDP per capita growth rate %
}

ALL_INDICATORS = {**TARGET_INDICATORS, **FEATURE_INDICATORS}

# DATE RANGE
START_YEAR = 2000
END_YEAR = 2023 # (last full year with reilaible data is 2022?)

# COUNTRY GROUPS — 44 countries
COUNTRIES = [
  # Asia
  "IN", "CN", "JP", "KR", "SG", "MY", "TH", "VN", "ID", "PH", 
  # Europe
  "DE", "FR", "GB", "ES", "IT", "SE", "NO", "DK", "NL", "PL",
  # Americas
  "US", "BR", "MX", "CA", "AR", "CL" ,"CO",
  # Africa & Middle East (full GCC: SA, AE, QA, KW, BH, OM)
  "ZA", "NG", "KE", "ET", "MA", "EG", "SA", "AE", "QA", "KW", "BH", "OM",
  # Others
  "AU", "NZ", "TR", "PK", "BD"
]

# The World Bank's official country names don't always match the display
# names used in load_country_metadata(). Any mismatch means the country is
# silently DROPPED during preprocessing (this cost us KR, VN, EG, AE, TR).
# Normalize immediately on load so every name matches the metadata exactly.
WB_NAME_FIXES = {
  "Korea, Rep.":           "South Korea",
  "Viet Nam":              "Vietnam",
  "Egypt, Arab Rep.":      "Egypt",
  "United Arab Emirates":  "UAE",
  "Turkiye":               "Turkey",
  "T\u00fcrkiye":          "Turkey",   # umlaut variant, just in case
}

CACHE_PATH = os.path.join(
  os.path.dirname(__file__), "..", "data", "cache", "wb_raw.pkl"
)
PROCESSED_PATH = os.path.join(
  os.path.dirname(__file__), "..", "data", "processed", "adoption_dataset.csv"
)

MAX_RETRIES = 3

# LOADER

def _normalize_country_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename World Bank official names to our metadata display names."""
    if "country" in df.columns:
        df = df.copy()
        df["country"] = df["country"].replace(WB_NAME_FIXES)
    return df


def _fetch_indicator_rest(code: str, col_name: str) -> pd.DataFrame:
    """
    Fetch one indicator straight from the World Bank REST API, bypassing wbdata.
    Returns a DataFrame indexed by (country, date) with a single value column —
    the same shape wbdata.get_dataframe produces, so results merge seamlessly.
    """
    import requests  # lazy import — only needed if wbdata fails

    url = (
        "https://api.worldbank.org/v2/country/"
        + ";".join(COUNTRIES)
        + f"/indicator/{code}"
    )

    rows = []
    page, total_pages = 1, 1
    while page <= total_pages:
        resp = requests.get(
            url,
            params={
                "format": "json",
                "per_page": 1000,
                "date": f"{START_YEAR}:{END_YEAR}",
                "page": page,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        if not isinstance(payload, list) or len(payload) < 2:
            raise RuntimeError(f"Unexpected API response for {code}: {payload}")

        meta, data = payload[0], payload[1] or []
        total_pages = int(meta.get("pages", 1))

        for d in data:
            rows.append({
                "country": d["country"]["value"],
                "date": d["date"],
                col_name: d["value"],
            })
        page += 1

    if not rows:
        raise RuntimeError(f"REST API returned no rows for {code}")

    df = pd.DataFrame(rows)
    return df.set_index(["country", "date"])


def fetch_world_bank_data(use_cache: bool = True) -> pd.DataFrame:

    # Downloads World Bank data for all indicators and countries.
    # Tries wbdata with retries, falls back to the REST API per indicator,
    # normalizes country names, and never caches an incomplete dataset.

    expected_cols = list(ALL_INDICATORS.values())

    if use_cache and os.path.exists(CACHE_PATH):
        print(f"Loading from cache: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            df = pickle.load(f)
        missing = [c for c in expected_cols if c not in df.columns]
        if not missing:
            # Normalize on load too, so an older cache self-heals
            return _normalize_country_names(df)
        print(f"Cache is incomplete — missing {missing}. Refetching from API.")

    all_dfs = []
    failed = []

    for code, col_name in ALL_INDICATORS.items():
        print(f"  Pulling: {col_name} ({code})")
        data = None

        # Primary path: wbdata, with retries
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                data = wbdata.get_dataframe(
                    {code: col_name},
                    country=COUNTRIES,
                )
                break
            except Exception as e:
                print(f"    Attempt {attempt}/{MAX_RETRIES} failed for {code}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 * attempt)

        # Fallback path: direct World Bank REST API
        if data is None:
            print(f"    wbdata failed — trying direct REST API for {code}")
            try:
                data = _fetch_indicator_rest(code, col_name)
                print(f"    REST fallback succeeded for {code}")
            except Exception as e:
                print(f"    REST fallback also failed for {code}: {e}")

        if data is None:
            failed.append(col_name)
        else:
            all_dfs.append(data)

    if failed:
        raise RuntimeError(
            f"Could not fetch (wbdata AND REST API both failed): {failed}. "
            "Nothing was cached — check the terminal above for the underlying error."
        )

    # Merge all indicators on country + date index
    df = all_dfs[0]
    for other in all_dfs[1:]:
        df = df.join(other, how="outer")

    df = df.reset_index()

    # wbdata returns multi-index (country, date) — normalize
    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df = df.drop(columns=["date"], errors="ignore")

    # Add country name from index if needed
    if "country" not in df.columns and df.index.name == "country":
        df = df.reset_index()


    df = _normalize_country_names(df)


    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Downloaded data is missing columns {missing}. Not caching."
        )


    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(df, f)
    print(f"\nCached to: {CACHE_PATH}")

    return df


def load_country_metadata() -> pd.DataFrame:
    metadata = {
        "country_code": [
            "IN","CN","JP","KR","SG","MY","TH","VN","ID","PH",
            "DE","FR","GB","ES","IT","SE","NO","DK","NL","PL",
            "US","BR","MX","CA","AR","CL","CO",
            "ZA","NG","KE","ET","MA","EG","SA","AE","QA","KW","BH","OM",
            "AU","NZ","TR","PK","BD",
        ],
        "country_name": [
            "India","China","Japan","South Korea","Singapore",
            "Malaysia","Thailand","Vietnam","Indonesia","Philippines",
            "Germany","France","United Kingdom","Spain","Italy",
            "Sweden","Norway","Denmark","Netherlands","Poland",
            "United States","Brazil","Mexico","Canada","Argentina",
            "Chile","Colombia",
            "South Africa","Nigeria","Kenya","Ethiopia","Morocco",
            "Egypt","Saudi Arabia","UAE","Qatar","Kuwait","Bahrain","Oman",
            "Australia","New Zealand","Turkey","Pakistan","Bangladesh",
        ],
        "region": [
            "South Asia","East Asia","East Asia","East Asia","East Asia",
            "Southeast Asia","Southeast Asia","Southeast Asia","Southeast Asia","Southeast Asia",
            "Europe","Europe","Europe","Europe","Europe",
            "Europe","Europe","Europe","Europe","Europe",
            "North America","Latin America","Latin America","North America","Latin America",
            "Latin America","Latin America",
            "Africa","Africa","Africa","Africa","Africa",
            "Middle East","Middle East","Middle East","Middle East","Middle East","Middle East","Middle East",
            "Oceania","Oceania","Europe","South Asia","South Asia",
        ],
        "income_group": [
            "Lower-middle","Upper-middle","High","High","High",
            "Upper-middle","Upper-middle","Lower-middle","Lower-middle","Lower-middle",
            "High","High","High","High","High",
            "High","High","High","High","High",
            "High","Upper-middle","Upper-middle","High","Upper-middle",
            "High","Upper-middle",
            "Upper-middle","Lower-middle","Lower-middle","Low","Lower-middle",
            "Lower-middle","High","High","High","High","High","High",
            "High","High","Upper-middle","Lower-middle","Lower-middle",
        ],
    }

    lengths = {k: len(v) for k, v in metadata.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Metadata lists are misaligned: {lengths}")

    missing = set(COUNTRIES) - set(metadata["country_code"])
    if missing:
        raise ValueError(f"Countries missing from metadata: {missing}")

    return pd.DataFrame(metadata)