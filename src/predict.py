#!/usr/bin/env python3
"""Payment Date Prediction — invoice payment-delay regression.

Rebuilds the documented HighRadius internship workflow (Payment Date
Prediction, project HRC80781W) as a runnable, leakage-free pipeline:

  1. Data inspection — shape, types, missing values, duplicates
  2. Data cleaning — drop redundant/unused columns, drop duplicates
  3. Standardization — parse dates, normalize USD/CAD amounts
     (CAD x 0.74 = USD, the conversion factor used by this rebuild)
  4. Dataset preparation — labeled rows (clearing date known) vs.
     unlabeled rows (clearing date to predict)
  5. Feature preparation — date components, categorical encoding,
     customer-level average payment delay
  6. Model comparison — Linear Regression, SVR, Decision Tree,
     Random Forest, XGBoost, all scored on the same held-out test set
  7. Prediction output — predicted delays, predicted payment dates
     and aging buckets for the unlabeled rows, exported as CSV

Correctness fix vs. the original notebook
-----------------------------------------
The original notebook computed customer-level target averages BEFORE
splitting the data, so test-row features leaked their own target
values and the reported scores were not reliable evidence of
predictive performance. This rebuild splits the labeled data into
train/test FIRST and computes every target-derived feature from the
training split only; unseen customers fall back to the training
global mean. All metrics below are measured on the held-out test set.

The bundled data is synthetic (see scripts/generate_sample_data.py).
Metrics are results on that synthetic sample, not real-world accuracy
or business-impact claims.

Usage:
    python src/predict.py --data data/invoices.csv --out outputs/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

CAD_TO_USD = 0.74
RANDOM_STATE = 42
TEST_SIZE = 0.20

DATE_COLS = [
    "clear_date",
    "posting_date",
    "document_create_date",
    "due_in_date",
    "baseline_create_date",
]
DROP_COLS = ["posting_id", "area_business", "is_open_item"]

NUMERIC_FEATURES = [
    "business_year",
    "terms_days",
    "amount_usd_log",
    "posting_month",
    "posting_dow",
    "due_month",
    "due_dow",
    "baseline_to_due_days",
    "cust_avg_delay",
    "cust_count",
]
CATEGORICAL_FEATURES = [
    "business_code",
    "document_type",
    "payment_method",
    "region",
]

AGING_BINS = [-np.inf, 0, 30, 60, 90, np.inf]
AGING_LABELS = [
    "Current",
    "1-30 days",
    "31-60 days",
    "61-90 days",
    "90+ days",
]


def load(path: Path) -> pd.DataFrame:
    # NOTE: the region code "NA" (North America) must not be read as
    # missing — pandas treats "NA" as a NaN marker by default. The
    # default NA set is therefore switched off and only empty strings
    # count as missing here.
    df = pd.read_csv(
        path,
        comment="#",
        parse_dates=DATE_COLS,
        keep_default_na=False,
        na_values=[""],
    )
    return df


def inspect(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "with_clear_date": int(df["clear_date"].notna().sum()),
        "without_clear_date": int(df["clear_date"].isna().sum()),
    }


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop redundant/unused columns and exact duplicate rows."""
    before = len(df)
    dropped_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=dropped_cols)
    df = df.drop_duplicates().reset_index(drop=True)
    info = {
        "dropped_columns": dropped_cols,
        "rows_before": before,
        "rows_after": len(df),
        "duplicates_removed": before - len(df),
    }
    return df, info


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, normalize amounts to USD, numeric payment terms."""
    df = df.copy()
    df["terms_days"] = (
        df["cust_payment_terms"].str.extract(r"(\d+)")[0].astype(float)
    )
    df["terms_days"] = df["terms_days"].fillna(df["terms_days"].mode()[0])
    df["amount_usd"] = np.where(
        df["invoice_currency"] == "CAD",
        df["amount_local"] * CAD_TO_USD,
        df["amount_local"],
    )
    df["amount_usd_log"] = np.log1p(df["amount_usd"])
    df["delay_days"] = (
        df["clear_date"] - df["due_in_date"]
    ).dt.days.astype("float")
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["posting_month"] = df["posting_date"].dt.month
    df["posting_dow"] = df["posting_date"].dt.day_of_week
    df["due_month"] = df["due_in_date"].dt.month
    df["due_dow"] = df["due_in_date"].dt.day_of_week
    df["baseline_to_due_days"] = (
        df["due_in_date"] - df["baseline_create_date"]
    ).dt.days
    return df


def add_customer_features(
    train: pd.DataFrame, *frames: pd.DataFrame
) -> list[pd.DataFrame]:
    """Customer-level delay stats computed from TRAIN ONLY.

    This is the leakage fix: the original notebook computed these
    averages before splitting, so test rows carried their own target
    in their features. Here the mapping is built on train and applied
    to every other frame; unseen customers get the train global mean.
    """
    stats = (
        train.groupby("cust_number")["delay_days"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "cust_avg_delay", "size": "cust_count"})
    )
    global_mean = float(train["delay_days"].mean())
    out = []
    for df in frames:
        df = df.copy()
        df["cust_avg_delay"] = df["cust_number"].map(stats["cust_avg_delay"])
        df["cust_count"] = df["cust_number"].map(stats["cust_count"])
        df["cust_avg_delay"] = df["cust_avg_delay"].fillna(global_mean)
        df["cust_count"] = df["cust_count"].fillna(0)
        out.append(df)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": Pipeline(
            [("scaler", StandardScaler()), ("svr", SVR())]
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_model_comparison(results: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = results.sort_values("rmse")
    bars = ax.barh(data["model"], data["rmse"], color="#1f77b4")
    ax.bar_label(bars, fmt="%.2f")
    ax.set_xlabel("Test RMSE (days)")
    ax.set_title("Model comparison — payment-delay prediction (held-out test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--")
    ax.set_xlabel("Actual delay (days)")
    ax.set_ylabel("Predicted delay (days)")
    ax.set_title("Actual vs. predicted payment delay (best model, test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delay_distribution(delay: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(delay, bins=60, kde=True, ax=ax, color="#1f77b4")
    ax.set_xlabel("Payment delay (days; negative = paid early)")
    ax.set_title("Distribution of payment delays (labeled rows)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_aging_buckets(pred: pd.DataFrame, path: Path) -> None:
    counts = pred["aging_bucket"].value_counts().reindex(AGING_LABELS).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color="#2ca02c")
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Invoices")
    ax.set_title("Predicted aging buckets — invoices without a clearing date")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    info: dict,
    cleaning: dict,
    results: pd.DataFrame,
    best_model: str,
    predictions: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []
    lines.append("# Payment Date Prediction — Report\n")
    lines.append(
        "> Results on the bundled **synthetic** invoice sample. "
        "Metrics are measured on the held-out test set; they are not "
        "real-world accuracy or business-impact claims.\n"
    )
    lines.append("## Data\n")
    lines.append(f"- Rows loaded: **{info['rows']:,}** x "
                 f"{info['columns']} columns")
    lines.append(f"- Duplicate rows found: **{info['duplicate_rows']:,}**")
    lines.append(f"- Rows with a clearing date: "
                 f"{info['with_clear_date']:,}")
    lines.append(f"- Rows without a clearing date: "
                 f"{info['without_clear_date']:,}")
    if info["missing"]:
        lines.append("- Missing values on load: " + ", ".join(
            f"{c} ({n:,})" for c, n in info["missing"].items()
        ))
    lines.append("")
    lines.append("## Cleaning\n")
    lines.append(f"- Dropped columns: "
                 f"{', '.join(cleaning['dropped_columns'])}")
    lines.append(f"- Rows after duplicate removal: "
                 f"**{cleaning['rows_after']:,}** "
                 f"({cleaning['duplicates_removed']:,} duplicates removed)")
    lines.append("")
    lines.append("## Leakage fix\n")
    lines.append(
        "The labeled data was split into train/test **before** any "
        "target-derived feature was computed. Customer-level average "
        "delay and invoice counts come from the training split only; "
        "unseen customers fall back to the training global mean. The "
        "original notebook computed these averages before splitting, "
        "which leaked each test row's own target into its features."
    )
    lines.append("")
    lines.append("## Model comparison (held-out test set)\n")
    lines.append("| Model | RMSE (days) | MAE (days) | R² |")
    lines.append("|-------|------------:|-----------:|---:|")
    for row in results.sort_values("rmse").itertuples():
        lines.append(
            f"| {row.model} | {row.rmse:.2f} | {row.mae:.2f} | {row.r2:.3f} |"
        )
    lines.append(f"\n**Selected model: {best_model}** "
                 "(lowest test RMSE).\n")
    lines.append("## Prediction output\n")
    lines.append(
        f"- Invoices without a clearing date: **{len(predictions):,}**"
    )
    lines.append(
        "- Predicted delay: "
        f"mean {predictions['predicted_delay_days'].mean():.1f} days, "
        f"median {predictions['predicted_delay_days'].median():.0f} days"
    )
    lines.append("- Aging buckets (predicted):")
    for label in AGING_LABELS:
        n = int((predictions["aging_bucket"] == label).sum())
        lines.append(f"  - {label}: {n:,}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Payment Date Prediction pipeline."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load(Path(args.data))
    info = inspect(raw)
    df, cleaning = clean(raw)
    df = standardize(df)
    df = add_date_features(df)

    labeled = df[df["clear_date"].notna()].reset_index(drop=True)
    unlabeled = df[df["clear_date"].isna()].reset_index(drop=True)
    print(f"Loaded: {info['rows']:,} rows x {info['columns']} columns")
    print(f"After cleaning: {len(df):,} rows "
          f"({cleaning['duplicates_removed']:,} duplicates removed)")
    print(f"Labeled (clearing date known): {len(labeled):,}")
    print(f"Unlabeled (to predict): {len(unlabeled):,}")

    train, test = train_test_split(
        labeled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train, test, unlabeled = add_customer_features(train, train, test, unlabeled)
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    results = []
    fitted = {}
    for name, model in get_models().items():
        pipe = Pipeline(
            [("prep", build_preprocessor()), ("model", model)]
        )
        pipe.fit(train[features], train["delay_days"])
        pred = pipe.predict(test[features])
        metrics = evaluate(test["delay_days"], pred)
        metrics["model"] = name
        results.append(metrics)
        fitted[name] = (pipe, pred)
        print(f"{name:28s} RMSE {metrics['rmse']:7.2f}  "
              f"MAE {metrics['mae']:6.2f}  R2 {metrics['r2']:6.3f}")

    results_df = pd.DataFrame(results)
    best_model = results_df.sort_values("rmse").iloc[0]["model"]
    best_pipe, best_test_pred = fitted[best_model]
    print(f"\nSelected model: {best_model}")

    unlabeled_pred = best_pipe.predict(unlabeled[features])
    predictions = pd.DataFrame(
        {
            "doc_id": unlabeled["doc_id"],
            "cust_number": unlabeled["cust_number"],
            "due_in_date": unlabeled["due_in_date"].dt.date,
            "predicted_delay_days": np.round(unlabeled_pred).astype(int),
        }
    )
    predictions["predicted_clear_date"] = (
        pd.to_datetime(predictions["due_in_date"])
        + pd.to_timedelta(predictions["predicted_delay_days"], unit="D")
    ).dt.date
    predictions["aging_bucket"] = pd.cut(
        predictions["predicted_delay_days"],
        bins=AGING_BINS,
        labels=AGING_LABELS,
    )
    predictions = predictions[
        [
            "doc_id",
            "cust_number",
            "due_in_date",
            "predicted_delay_days",
            "predicted_clear_date",
            "aging_bucket",
        ]
    ]
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    plot_model_comparison(results_df, out_dir / "model_comparison.png")
    plot_actual_vs_predicted(
        test["delay_days"], best_test_pred,
        out_dir / "actual_vs_predicted.png",
    )
    plot_delay_distribution(
        labeled["delay_days"], out_dir / "delay_distribution.png"
    )
    plot_aging_buckets(predictions, out_dir / "aging_buckets.png")
    write_report(
        info, cleaning, results_df, best_model, predictions,
        out_dir / "report.md",
    )
    print(f"\nWrote: {out_dir / 'predictions.csv'} ({len(predictions):,} rows)")
    print(f"Wrote: {out_dir / 'report.md'}")
    print("Wrote: model_comparison.png, actual_vs_predicted.png, "
          "delay_distribution.png, aging_buckets.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
