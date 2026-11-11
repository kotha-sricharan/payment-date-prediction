# Payment Date Prediction — Report

> Results on the bundled **synthetic** invoice sample. Metrics are measured on the held-out test set; they are not real-world accuracy or business-impact claims.

## Data

- Rows loaded: **50,000** x 19 columns
- Duplicate rows found: **2,308**
- Rows with a clearing date: 40,185
- Rows without a clearing date: 9,815
- Missing values on load: name_customer (231), clear_date (9,815), cust_payment_terms (764)

## Cleaning

- Dropped columns: posting_id, area_business, is_open_item
- Rows after duplicate removal: **47,692** (2,308 duplicates removed)

## Leakage fix

The labeled data was split into train/test **before** any target-derived feature was computed. Customer-level average delay and invoice counts come from the training split only; unseen customers fall back to the training global mean. The original notebook computed these averages before splitting, which leaked each test row's own target into its features.

## Model comparison (held-out test set)

| Model | RMSE (days) | MAE (days) | R² |
|-------|------------:|-----------:|---:|
| Linear Regression | 12.12 | 9.71 | 0.698 |
| XGBoost | 12.27 | 9.84 | 0.690 |
| Support Vector Regression | 12.51 | 10.01 | 0.678 |
| Random Forest | 12.52 | 10.05 | 0.677 |
| Decision Tree | 17.79 | 14.27 | 0.349 |

**Selected model: Linear Regression** (lowest test RMSE).

## Prediction output

- Invoices without a clearing date: **9,367**
- Predicted delay: mean 15.1 days, median 12 days
- Aging buckets (predicted):
  - Current: 2,290
  - 1-30 days: 5,222
  - 31-60 days: 1,686
  - 61-90 days: 169
  - 90+ days: 0
