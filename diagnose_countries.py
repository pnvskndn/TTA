
import os
import sys
import ast
import pickle
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from data_loader import COUNTRIES

RAW_PATH = os.path.join(ROOT, "data", "cache", "wb_raw.pkl")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "adoption_dataset.csv")
PREPROCESSOR_SRC = os.path.join(ROOT, "src", "preprocessor.py")

BAR = "=" * 62


print(BAR)
print("STAGE 1 — RAW DOWNLOAD  (data/cache/wb_raw.pkl)")
print(BAR)

raw_names = None
if os.path.exists(RAW_PATH):
    with open(RAW_PATH, "rb") as f:
        raw = pickle.load(f)

    if "country" in raw.columns:
        name_col = "country"
    else:
        obj_cols = [c for c in raw.columns if raw[c].dtype == object]
        name_col = obj_cols[0] if obj_cols else None

    if name_col:
        raw_names = sorted(raw[name_col].astype(str).unique())
        print(f"Unique country names in raw download: {len(raw_names)}")
        for n in raw_names:
            print(f"    {n}")
    else:
        print("Could not find a country-name column. Columns are:")
        print("   ", list(raw.columns))
else:
    print(f"No raw cache found at {RAW_PATH}")

print()


print(BAR)
print("STAGE 2 — NAME→ISO2 MAPPING  (src/preprocessor.py)")
print(BAR)

mapping = {}
try:
    tree = ast.parse(open(PREPROCESSOR_SRC).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            try:
                d = ast.literal_eval(node)
            except Exception:
                continue
            if d and all(
                isinstance(k, str) and isinstance(v, str)
                and len(v) == 2 and v.isupper()
                for k, v in d.items()
            ):
                mapping.update(d)

    print(f"Name→code entries found: {len(mapping)}")

    codes_in_mapping = set(mapping.values())
    missing_codes = [c for c in COUNTRIES if c not in codes_in_mapping]
    print(f"In COUNTRIES but ABSENT from the mapping: "
          f"{', '.join(missing_codes) if missing_codes else 'none'}")

    if raw_names is not None:
        unmapped = [n for n in raw_names if n not in mapping]
        if unmapped:
            print("Downloaded names the mapping does NOT recognise "
                  "(these rows get DROPPED in clean):")
            for n in unmapped:
                print(f"    {n}")
        else:
            print("Every downloaded country name maps successfully.")
except FileNotFoundError:
    print(f"Could not find {PREPROCESSOR_SRC}")

print()


print(BAR)
print("STAGE 3 — PROCESSED DATASET  (data/processed/adoption_dataset.csv)")
print(BAR)

if os.path.exists(PROCESSED_PATH):
    proc = pd.read_csv(PROCESSED_PATH)
    codes = sorted(proc["country_code"].dropna().astype(str).unique())
    print(f"Countries in processed dataset: {len(codes)}")
    print("   ", ", ".join(codes))

    lost = [c for c in COUNTRIES if c not in codes]
    print(f"Expected but MISSING: {', '.join(lost) if lost else 'none'}")

    print()
    print("Rows and year coverage per country:")
    cov = (
        proc.dropna(subset=["year"])
        .groupby("country_code")["year"]
        .agg(rows="count", first_year="min", last_year="max")
        .astype(int)
    )
    print(cov.to_string())
else:
    print(f"No processed CSV found at {PROCESSED_PATH}")

print()
