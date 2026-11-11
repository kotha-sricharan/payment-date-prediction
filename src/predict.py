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
     Random Forest, XGBoost, compared on a validation split
  7. Final evaluation — the winning model is refit on train +
     validation and scored exactly once on an untouched test set
  8. Prediction output — predicted delays, predicted payment dates
     and aging buckets for the unlabeled rows, exported as CSV

Correctness fixes vs. the original notebook
--------------------------------------------
The original notebook computed customer-level target averages BEFORE
splitting the data, so test-row features leaked their own target
values and the reported scores were not reliable evidence of
predictive performance. This rebuild fixes three evaluation issues:

1. Target leakage — the labeled data is split into train /
   validation / test FIRST, and customer-level features are computed
   from the training split only (unseen customers fall back to the
   training global mean).
2. Imputation leakage — missing payment terms are filled with the
   most common term computed from the TRAINING split only, then
   applied to validation, test, and unlabeled rows. (The pipeline's
   own imputers and scaler are fit on training data only, as
   scikit-learn Pipelines guarantee.)
3. Selection on test — models are compared on the VALIDATION split.
   The winner is refit on train + validation and evaluated exactly
   once on the untouched test set, so the reported test metrics are
   not the numbers the selection was tuned against.

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
    """Parse dates, normalize amounts to USD, numeric payment terms.

    Missing payment terms are NOT filled here: the fill value (the
    most common term) is a dataset statistic, so it is computed from
    the training split only, after splitting — see
    fill_terms_days_mode(). Filling it here would leak information
    from validation/test rows into training.
    """
    df = df.copy()
    df["terms_days"] = (
        df["cust_payment_terms"].str.extract(r"(\d+)")[0].astype(float)
    )
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


def fill_terms_days_mode(
    train: pd.DataFrame, *frames: pd.DataFrame
) -> list[pd.DataFrame]:
    """Fill missing payment terms with the TRAINING mode only.

    The most common term is computed from the training frame and then
    applied to every other frame, so no information flows from
    validation/test rows into training.
    """
    mode = float(train["terms_days"].mode()[0])
    out = []
    for df in frames:
        df = df.copy()
        df["terms_days"] = df["terms_days"].fillna(mode)
        out.append(df)
    return out


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
    ax.set_xlabel("Validation RMSE (days)")
    ax.set_title("Model comparison — payment-delay prediction (validation set)")
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
    test_metrics: dict,
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
    lines.append("## Evaluation design (leakage fixes)\n")
    lines.append(
        "1. The labeled data was split into train / validation / test "
        "**before** any target-derived feature was computed. "
        "Customer-level average delay and invoice counts come from "
        "the training split only; unseen customers fall back to the "
        "training global mean. The original notebook computed these "
        "averages before splitting, which leaked each test row's own "
        "target into its features."
    )
    lines.append(
        "2. Missing payment terms are filled with the most common "
        "term computed from the **training** split only, then applied "
        "to validation, test, and unlabeled rows."
    )
    lines.append(
        "3. Models were compared on the **validation** split. The "
        "winner was refit on train + validation and evaluated exactly "
        "once on the untouched test set, so the reported test metrics "
        "are not the numbers the selection was tuned against."
    )
    lines.append("")
    lines.append("## Model comparison (validation set)\n")
    lines.append("| Model | Validation RMSE (days) | Validation MAE (days) | Validation R² |")
    lines.append("|-------|-----------------------:|----------------------:|--------------:|")
    for row in results.sort_values("rmse").itertuples():
        lines.append(
            f"| {row.model} | {row.rmse:.2f} | {row.mae:.2f} | {row.r2:.3f} |"
        )
    lines.append(f"\n**Selected model: {best_model}** "
                 "(lowest validation RMSE).\n")
    lines.append("## Final evaluation (untouched test set, evaluated once)\n")
    lines.append(
        "The selected model was refit on train + validation and scored "
        "a single time on the test set, which played no part in "
        "training or model selection."
    )
    lines.append(f"- Model: **{best_model}**")
    lines.append(f"- Test RMSE: **{test_metrics['rmse']:.2f} days**")
    lines.append(f"- Test MAE: **{test_metrics['mae']:.2f} days**")
    lines.append(f"- Test R²: **{test_metrics['r2']:.3f}**")
    lines.append("")
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

    # Labeled data: 60% train / 20% validation / 20% test.
    # The test set is held out from the start and touched exactly
    # once, for the final evaluation — it plays no part in training,
    # imputation, feature building, or model selection.
    train_val, test = train_test_split(
        labeled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        train_val, test_size=0.25, random_state=RANDOM_STATE
    )
    print(f"Train: {len(train):,}  Validation: {len(val):,}  "
          f"Test: {len(test):,} (untouched until final evaluation)")

    # --- selection phase: every train-only statistic comes from `train`
    train, val = fill_terms_days_mode(train, train, val)
    train, val = add_customer_features(train, train, val)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    val_results = []
    for name, model in get_models().items():
        pipe = Pipeline(
            [("prep", build_preprocessor()), ("model", model)]
        )
        pipe.fit(train[features], train["delay_days"])
        pred = pipe.predict(val[features])
        metrics = evaluate(val["delay_days"], pred)
        metrics["model"] = name
        val_results.append(metrics)
        print(f"{name:28s} val RMSE {metrics['rmse']:7.2f}  "
              f"MAE {metrics['mae']:6.2f}  R2 {metrics['r2']:6.3f}")

    val_df = pd.DataFrame(val_results)
    best_model = val_df.sort_values("rmse").iloc[0]["model"]
    print(f"\nSelected model (lowest validation RMSE): {best_model}")

    # --- final phase: refit the winner on train + validation, then
    # evaluate ONCE on the untouched test set ---
    train_val, test, unlabeled = fill_terms_days_mode(
        train_val, train_val, test, unlabeled
    )
    train_val, test, unlabeled = add_customer_features(
        train_val, train_val, test, unlabeled
    )
    best_pipe = Pipeline(
        [("prep", build_preprocessor()),
         ("model", get_models()[best_model])]
    )
    best_pipe.fit(train_val[features], train_val["delay_days"])
    test_pred = best_pipe.predict(test[features])
    test_metrics = evaluate(test["delay_days"], test_pred)
    print(f"Final test evaluation ({best_model}): "
          f"RMSE {test_metrics['rmse']:.2f}  "
          f"MAE {test_metrics['mae']:.2f}  "
          f"R2 {test_metrics['r2']:.3f}")

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

    plot_model_comparison(val_df, out_dir / "model_comparison.png")
    plot_actual_vs_predicted(
        test["delay_days"], test_pred,
        out_dir / "actual_vs_predicted.png",
    )
    plot_delay_distribution(
        labeled["delay_days"], out_dir / "delay_distribution.png"
    )
    plot_aging_buckets(predictions, out_dir / "aging_buckets.png")
    write_report(
        info, cleaning, val_df, best_model, test_metrics, predictions,
        out_dir / "report.md",
    )
    print(f"\nWrote: {out_dir / 'predictions.csv'} ({len(predictions):,} rows)")
    print(f"Wrote: {out_dir / 'report.md'}")
    print("Wrote: model_comparison.png, actual_vs_predicted.png, "
          "delay_distribution.png, aging_buckets.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
