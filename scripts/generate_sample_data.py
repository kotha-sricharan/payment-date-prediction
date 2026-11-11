#!/usr/bin/env python3
"""Generate the synthetic invoice dataset for payment-date-prediction.

Reproduces the documented shape of the HighRadius internship workflow:
50,000 rows and 19 columns on load, 47,692 rows after duplicate
removal, of which 38,325 carry a payment-clearing date and 9,367 do
not. The counts are engineered exactly (47,692 unique invoices plus
2,308 exact duplicate rows), so the pipeline's inspection step lands
on the documented numbers.

All data is synthetic (fixed seed 42). Customer names, amounts, dates
and delays are invented for demonstration; nothing comes from a real
invoice ledger.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --out data/invoices.csv --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_UNIQUE = 47_692
N_DUPLICATES = 2_308
N_LABELED = 38_325
CAD_SHARE = 0.15
CAD_TO_USD = 0.74  # conversion factor applied by the pipeline

CUSTOMER_NAMES = [
    "Acme Distribution", "BluePeak Foods", "Cascade Retail Group",
    "Delta Manufacturing", "Everline Logistics", "Frontier Health",
    "Golden Gate Apparel", "Harbor Tech Solutions", "Ironwood Paper",
    "Juniper Energy", "Keystone Automotive", "Lakeside Beverages",
    "Meridian Pharmaceuticals", "Northgate Electronics", "Oakridge Foods",
    "Pinnacle Textiles", "Quantum Chemicals", "Riverbend Plastics",
    "Summit Building Supplies", "TrueNorth Media",
]

TERMS = [15, 30, 45, 60, 90]
TERMS_WEIGHTS = [0.10, 0.45, 0.25, 0.15, 0.05]
DOC_TYPES = ["RV", "AB", "DZ", "KR"]
DOC_WEIGHTS = [0.80, 0.08, 0.07, 0.05]
PAYMENT_METHODS = ["ACH", "Wire", "Check", "Card"]
METHOD_WEIGHTS = [0.55, 0.25, 0.12, 0.08]
REGIONS = ["NA", "EU", "APAC"]
REGION_WEIGHTS = [0.70, 0.20, 0.10]
BUSINESS_CODES = ["U001", "U002"]
BUSINESS_WEIGHTS = [0.70, 0.30]


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- customers: each has its own typical payment-delay profile ---
    n_customers = 200
    cust_ids = [f"C{1001 + i}" for i in range(n_customers)]
    cust_names = rng.choice(CUSTOMER_NAMES, size=n_customers)
    cust_base_delay = np.clip(rng.normal(15, 20, n_customers), -10, 70)
    cust_amount_scale = np.exp(rng.normal(9.5, 0.8, n_customers))
    cust_region = rng.choice(REGIONS, size=n_customers, p=REGION_WEIGHTS)

    cust_idx = rng.integers(0, n_customers, N_UNIQUE)
    cust_number = np.array(cust_ids)[cust_idx]
    name_customer = np.array(cust_names)[cust_idx]
    base_delay = cust_base_delay[cust_idx]
    amount_scale = cust_amount_scale[cust_idx]
    region = np.array(cust_region)[cust_idx]

    # --- dates ---
    baseline = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 730, N_UNIQUE), unit="D"
    )
    posting_date = baseline + pd.to_timedelta(rng.integers(0, 3, N_UNIQUE), unit="D")
    terms = rng.choice(TERMS, size=N_UNIQUE, p=TERMS_WEIGHTS)
    due_in_date = baseline + pd.to_timedelta(terms, unit="D")

    # --- payment delay: customer profile + noise; negative = paid early ---
    delay = np.round(base_delay + rng.normal(0, 12, N_UNIQUE)).astype(int)
    clear_date = due_in_date + pd.to_timedelta(delay, unit="D")

    # --- labeled / unlabeled split: exactly 38,325 with a clearing date ---
    labeled = np.zeros(N_UNIQUE, dtype=bool)
    labeled[rng.choice(N_UNIQUE, N_LABELED, replace=False)] = True
    clear_date = pd.Series(clear_date)
    clear_date[~labeled] = pd.NaT

    # --- amounts: 85% USD, 15% CAD (pipeline converts CAD at 0.74) ---
    currency = np.where(rng.random(N_UNIQUE) < CAD_SHARE, "CAD", "USD")
    amount_local = np.round(
        rng.lognormal(np.log(amount_scale), 0.9), 2
    )

    # --- remaining columns ---
    business_code = rng.choice(BUSINESS_CODES, size=N_UNIQUE, p=BUSINESS_WEIGHTS)
    business_year = baseline.year.values
    doc_id = 19_000_000_00 + np.arange(N_UNIQUE)
    document_type = rng.choice(DOC_TYPES, size=N_UNIQUE, p=DOC_WEIGHTS)
    payment_method = rng.choice(
        PAYMENT_METHODS, size=N_UNIQUE, p=METHOD_WEIGHTS
    )
    is_open_item = (~labeled).astype(int)
    posting_id = np.arange(1, N_UNIQUE + 1)
    area_business = np.array(["B2B"] * N_UNIQUE)

    df = pd.DataFrame(
        {
            "business_code": business_code,
            "cust_number": cust_number,
            "name_customer": name_customer,
            "clear_date": clear_date,
            "business_year": business_year,
            "doc_id": doc_id,
            "posting_date": posting_date,
            "document_create_date": baseline,
            "due_in_date": due_in_date,
            "baseline_create_date": baseline,
            "cust_payment_terms": [f"N{t}" for t in terms],
            "amount_local": amount_local,
            "invoice_currency": currency,
            "document_type": document_type,
            "posting_id": posting_id,
            "area_business": area_business,
            "is_open_item": is_open_item,
            "payment_method": payment_method,
            "region": region,
        }
    )

    # --- realistic data issues: a few missing values ---
    terms_missing = rng.random(N_UNIQUE) < 0.015
    df.loc[terms_missing, "cust_payment_terms"] = np.nan
    name_missing = rng.random(N_UNIQUE) < 0.005
    df.loc[name_missing, "name_customer"] = np.nan

    # --- exact duplicates: 2,308 copied rows -> 50,000 rows on load ---
    dups = df.iloc[rng.choice(N_UNIQUE, N_DUPLICATES, replace=False)].copy()
    df = pd.concat([df, dups], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic invoice dataset."
    )
    parser.add_argument("--out", default="data/invoices.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# SYNTHETIC DATA — generated by scripts/generate_sample_data.py "
        "(fixed seed). Not real invoice data.\n"
    )
    with out.open("w", encoding="utf-8") as f:
        f.write(header)
        df.to_csv(f, index=False)
    print(f"Wrote {len(df):,} rows x {len(df.columns)} columns to {out}")
    print(f"Rows with clear_date: {df['clear_date'].notna().sum():,}")
    print(f"Rows without clear_date: {df['clear_date'].isna().sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
