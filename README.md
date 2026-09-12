# Fraud Detection PaySim Synthetic Financial Dataset

A fraud-detection project on the [PaySim synthetic mobile money dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) (~6.3M transactions, ~0.13% fraud), covering data cleaning, feature engineering, extreme class imbalance handling, and a comparison of three model types including a real investigation into a data-leakage red flag along the way.

## What This Covers

- **Parts 1-4:** Data exploration, quality checks, and identifying that fraud is structurally confined to `CASH_OUT` and `TRANSFER` transactions
- **Part 5:** Feature engineering — two balance-discrepancy features that capture a counterintuitive fraud signal
- **Parts 6-8:** Data prep, stratified train/test split, and a Logistic Regression baseline
- **Parts 9-11:** Random Forest, an investigation into a suspiciously perfect result (data leakage), and a corrected retrain
- **Part 12:** GPU-accelerated XGBoost as a third model, using CUDA

## Key Findings

- **Fraud only occurs in 2 of 5 transaction types** (`CASH_OUT`, `TRANSFER`) — filtering to these cut the dataset by more than half with zero loss of fraud cases.
- **Engineered balance-discrepancy features revealed a counterintuitive pattern:** legitimate transactions have messy, inconsistent balances, while fraud transactions are suspiciously "clean" — money seems to leave the sender properly but not fully arrive at the receiver.
- **A near-perfect Random Forest result (100% precision/recall) was investigated rather than reported at face value.** `feature_importances_` showed one engineered feature accounted for 42% of the model's decisions — traced to a likely artifact of the synthetic data generation process rather than a real fraud pattern. Retraining without it produced an honest, defensible result (94% precision, 82% recall).
- **Random Forest and XGBoost represent a genuine business tradeoff:** Random Forest gives far fewer false alarms (88 vs. 1,419) but misses more fraud; XGBoost catches nearly all fraud (98% recall) at a much higher false-alarm cost. The right choice depends on whether an organization prioritizes review workload or missed-fraud risk.

## Model Comparison

| Model | Precision (fraud) | Recall (fraud) | False Positives | False Negatives | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.02 | 0.95 | 62,728 | 81 | 0.974 |
| Random Forest (with leaky feature) | 1.00 | 1.00 | 1 | 5 | 0.999 |
| Random Forest (honest) | 0.94 | 0.82 | 88 | 292 | 0.997 |
| XGBoost (honest, GPU) | 0.53 | 0.98 | 1,419 | 26 | 0.999 |

## Tools

- `pandas`, `numpy` — data manipulation
- `scikit-learn` — Logistic Regression, Random Forest, evaluation metrics
- `xgboost` — GPU-accelerated gradient boosting (trained via CUDA)

## Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) — file is `PS_20174392719_1491204439457_log.csv`.
2. `pip install pandas numpy matplotlib seaborn scikit-learn xgboost`
3. Place the CSV in the same folder as the notebook and run all cells.
   - Note: the notebook trains XGBoost with `device='cuda'`, which requires an NVIDIA GPU with CUDA support. To run on CPU instead, remove the `device='cuda'` and `tree_method='hist'` arguments from the `XGBClassifier` call.

## Dataset

[PaySim — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) — ~6.3M simulated mobile money transactions with realistic transaction types, amounts, and account balances, based on real transaction logs from a mobile money service.
