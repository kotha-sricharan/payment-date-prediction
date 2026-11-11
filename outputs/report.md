# Payment Date Prediction — Report

> Results from this portfolio rebuild on the bundled **synthetic** invoice sample — **not** the original internship's results. Metrics are measured on the held-out test set; they are not real-world accuracy or business-impact claims.

## Data

- Rows loaded: **50,000** x 19 columns
- Duplicate rows found: **2,308**
- Rows with a clearing date: 40,185
- Rows without a clearing date: 9,815
- Missing values on load: name_customer (231), clear_date (9,815), cust_payment_terms (764)

## Cleaning

- Dropped columns: posting_id, area_business, is_open_item
- Rows after duplicate removal: **47,692** (2,308 duplicates removed)

## Evaluation design (leakage fixes)

1. The labeled data was split into train / validation / test **before** any target-derived feature was computed. Customer-level average delay and invoice counts come from the training split only; unseen customers fall back to the training global mean. The original notebook computed these averages before splitting, which leaked each test row's own target into its features.
2. Missing payment terms are filled with the most common term computed from the **training** split only, then applied to validation, test, and unlabeled rows.
3. Models were compared on the **validation** split. The winner was refit on train + validation and evaluated exactly once on the untouched test set, so the reported test metrics are not the numbers the selection was tuned against.

## Model comparison (validation set)

| Model | Validation RMSE (days) | Validation MAE (days) | Validation R² |
|-------|-----------------------:|----------------------:|--------------:|
| Linear Regression | 11.96 | 9.53 | 0.709 |
| XGBoost | 12.12 | 9.67 | 0.701 |
| Random Forest | 12.32 | 9.81 | 0.691 |
| Support Vector Regression | 12.52 | 9.97 | 0.681 |
| Decision Tree | 17.67 | 14.07 | 0.365 |

**Selected model: Linear Regression** (lowest validation RMSE).

## Final evaluation (untouched test set, evaluated once)

The selected model was refit on train + validation and scored a single time on the test set, which played no part in training or model selection.
- Model: **Linear Regression**
- Test RMSE: **12.12 days**
- Test MAE: **9.71 days**
- Test R²: **0.698**

## Prediction output

- Invoices without a clearing date: **9,367**
- Predicted delay: mean 15.1 days, median 12 days
- Aging buckets (predicted):
  - Current: 2,290
  - 1-30 days: 5,222
  - 31-60 days: 1,686
  - 61-90 days: 169
  - 90+ days: 0
