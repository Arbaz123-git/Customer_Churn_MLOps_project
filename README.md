# Telco Customer Churn

Predicts customer churn for a telecom provider. The approved model is a
hyperparameter-tuned XGBoost classifier (recall 0.922, ROC-AUC 0.846 on the
held-out test set), originally found via Optuna search in
`notebook/eda.ipynb` and now reproduced deterministically by the `churn`
package.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Train

```
churn-train
```

This loads `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, cleans it, splits
train/test, fits the feature encoder + XGBoost pipeline using the frozen
hyperparameters in `config/config.yaml`, evaluates on the test set, checks
the result against the approved run's metrics (logged as a `parity` tag),
and logs params/metrics/the model to MLflow, registering it as
`telco-churn-xgb`.

Options:

- `--config PATH` — use a different config file (default `config/config.yaml`)
- `--no-register` — log the run without registering a new model version

## Inspect runs

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Load the trained model

```python
import mlflow

model = mlflow.sklearn.load_model("models:/telco-churn-xgb/latest")
model.predict_proba(raw_customer_rows)  # raw columns, no manual encoding needed
```

## Layout

- `src/churn/` — the package: `config.py` (load `config/config.yaml`),
  `data.py` (load/clean/split), `features.py` (`TelcoFeatureEncoder`,
  reproduces the notebook's binary-map + one-hot encoding as a fittable
  sklearn transformer), `model.py` (builds the sklearn `Pipeline`),
  `evaluate.py` (metrics + threshold sweep), `train.py` (CLI entry point).
- `config/config.yaml` — column lists, split settings, the frozen approved
  hyperparameters, MLflow settings, and the reference metrics used for the
  parity check.
- `notebook/eda.ipynb` — the original exploration: EDA, correlation/VIF
  analysis, model comparison (RandomForest/LightGBM/XGBoost), and the
  Optuna search that produced the approved hyperparameters. Kept as a
  historical record; not used by the pipeline.
