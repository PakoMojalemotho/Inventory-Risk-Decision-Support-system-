# Explainable, Risk-Based Inventory Decision Support — Training BOP

A VS Code-ready Python prototype for the project's **actual Training BOP (backorder prediction) dataset**. It trains and compares classifiers, evaluates the result with imbalance-aware metrics, exports risk-scored CSV files, and provides a Streamlit decision-support dashboard with transparent inventory recommendations.

> **Important target caveat:** The known BOP target `went_on_backorder` represents the backorder label in that dataset. It does not on its own prove a 14–30-day future stockout horizon. The standard cross-sectional table also does not, by itself, support training a separate time-series demand forecaster; the dashboard uses existing sales/forecast fields for an approximate stock-cover heuristic. The system clearly documents these limitations instead of claiming unsupported forecasts. Read `docs/DATASET_AND_TARGET.md`.

## 1. Put your cleaned data in the project

Copy the cleaned CSV your team actually used into:

```text
data/Training_BOP.csv
```

The dataset is deliberately not included in this ZIP. You do not need to upload a large dataset to start locally; keep it on your computer and use the real path if it is stored elsewhere.

The target defaults to `went_on_backorder`. Column case and spaces are normalised. If your cleaned target has another name, use `--target your_column_name`.

## 2. Set up in VS Code (Windows PowerShell)

Open this folder in VS Code, then select **Terminal → New Terminal** and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To run the included unit tests (optional):

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

If PowerShell blocks virtual-environment activation, run the commands through `\.venv\Scripts\python.exe` directly instead of changing system policy.

## 3. Inspect the cleaned data

```powershell
python src/eda.py --data data/Training_BOP.csv
```

Outputs appear under `reports/`: `data_profile.csv`, `data_profile_summary.json`, and `target_distribution.png`. Review data quality, label balance, target meaning, and any unexpected columns before modelling.

## 4. Train and compare the models

```powershell
python train_model.py --data data/Training_BOP_Cleaned.csv
```

By default, the script caps local training at 150,000 rows for practical development on a laptop and samples the training partition stratified by target. Use the full training partition if your computer can handle it:

```powershell
python train_model.py --data data/Training_BOP_Cleaned.csv --max-train-rows 0
```

If your target column was renamed:

```powershell
python train_model.py --data data/Training_BOP_Cleaned.csv --target went_on_backorder
```

Candidate models:

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting

The selected model is the strongest on validation F2 (which weights recall more heavily because missed backorders matter). The chosen threshold is selected on validation data, then held fixed for the final test metrics. If your CSV contains a `Validation` split coded `0/1/2`, the code uses it as train/validation/test when all partitions contain both classes; otherwise it creates a stratified 70%/15%/15% split.

Generated artefacts:

- `models/best_model.joblib` — fitted preprocessing + selected classifier + configuration
- `reports/metrics.json` — model comparison and final test metrics
- `reports/feature_importance.csv` — global permutation importance by source feature
- `reports/test_predictions.csv` — test-set probabilities and predictions

## 5. Launch the dashboard

```powershell
streamlit run app.py
```

Upload a cleaned/current inventory CSV and the system will provide risk categories, a monthly demand proxy and estimated days of cover where BOP-style sales/inventory columns are available, action recommendations, row-level operational checks, model evaluation, and a CSV export. It does not place orders or connect to a live database.

Optional local SHAP explanations can be enabled with:

```powershell
pip install shap
```

SHAP support depends on the chosen estimator and package compatibility; the dashboard has a fallback if it cannot generate a local explanation.

## 6. Batch-score another file from the terminal

```powershell
python src/predict.py --input data/current_inventory.csv --output reports/scored_inventory.csv
```

Optional risk-band adjustments:

```powershell
python src/predict.py --input data/current_inventory.csv --output reports/scored_inventory.csv --medium-threshold 0.25 --high-threshold 0.60
```

The scoring file can omit the target column. Missing model features are imputed, but a different schema, unit, or meaning may make results unreliable. Keep the training and scoring feature definitions consistent.

## 7. Project structure

```text
inventory_bop_system/
├── app.py
├── requirements.txt
├── data/                 # put the cleaned Training_BOP.csv here
├── models/               # trained model is saved here
├── reports/              # EDA, metrics, predictions and importance
├── src/
│   ├── common.py
│   ├── decision_logic.py
│   ├── eda.py
│   ├── predict.py
│   └── train.py
├── docs/
│   ├── DATASET_AND_TARGET.md
│   └── SYSTEM_ARCHITECTURE.md
├── tests/
└── .vscode/
```

## 8. Before using results in the final project report

1. Confirm that the cleaned dataset contains the fields the team actually used and that `went_on_backorder` remains the correct label.
2. Inspect target imbalance, missingness, possible leakage, validation split semantics, and class counts.
3. Report the final held-out test metrics from `reports/metrics.json`; do not invent or pre-state accuracy.
4. Document the selected threshold, false negatives, false positives, and how inventory managers should interpret the alert bands.
5. Do not describe the output as a 14–30-day prediction unless time-stamped future-outcome data can validate that prediction window.

For architecture and the target caveat, see `docs/SYSTEM_ARCHITECTURE.md` and `docs/DATASET_AND_TARGET.md`.


## Model-specific notes (updated)

- The dashboard loads the saved bundle at `models/best_model.joblib`; uploading a CSV scores it and does not retrain the model.
- Training uses class weighting for imbalanced labels and a fixed positive-class probability threshold of `0.10`. Review precision, recall, false negatives, and alert volume in `reports/metrics.json` before treating this as a production setting.
- Deterministic BOP engineered features are created by the shared `normalise_dataframe` function, so training and scoring use the same calculations.
- The optional SHAP local explanation requires `pip install shap`; install the packages in `requirements.txt` first.
- The reorder quantity is explicitly labelled as assuming `lead_time` is in DAYS. The notebook available for review does not establish the unit. If it is in weeks, convert to days (multiply by 7) before relying on the suggested quantity. `min_bank` is only a simple buffer proxy here, not a validated 20% safety-stock policy.
- Negative stock cover is retained when inventory position is negative; interpret it as a deficit signal, not a literal negative number of days until stockout.
- The available notebook contains EDA and cleaning steps but no final fitted-model training cells, so this package is a best-effort reproducible implementation, not a verified copy of an unseen notebook model artifact.
