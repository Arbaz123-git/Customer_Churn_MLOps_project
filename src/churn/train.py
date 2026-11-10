from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from churn.config import Config, DEFAULT_CONFIG_PATH, load_config
from churn.data import clean, load_raw, split_xy
from churn.evaluate import compute_metrics, threshold_sweep
from churn.model import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("churn.train")


def run(cfg: Config, register: bool = True) -> dict[str, float]:
    df = clean(load_raw(cfg.raw_path), cfg)
    X_train, X_test, y_train, y_test = split_xy(df, cfg)

    pipeline = build_pipeline(
        cfg.model_params, y_train, cfg.features.binary, cfg.features.categorical
    )
    scale_pos_weight = pipeline.named_steps["model"].get_params()["scale_pos_weight"]

    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_train

    start_pred = time.time()
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start_pred

    metrics = compute_metrics(y_test, proba, cfg.threshold)
    sweep = threshold_sweep(y_test, proba)

    logger.info("recall=%.4f precision=%.4f f1=%.4f roc_auc=%.4f",
                metrics["recall"], metrics["precision"], metrics["f1"], metrics["roc_auc"])

    parity_ok = (
        abs(metrics["recall"] - cfg.reference.recall) <= cfg.reference.tolerance
        and abs(metrics["roc_auc"] - cfg.reference.roc_auc) <= cfg.reference.tolerance
    )
    if not parity_ok:
        logger.warning(
            "Metrics diverge from approved run %s by more than tolerance %.3f "
            "(recall %.4f vs %.4f, roc_auc %.4f vs %.4f)",
            cfg.reference.run_id,
            cfg.reference.tolerance,
            metrics["recall"],
            cfg.reference.recall,
            metrics["roc_auc"],
            cfg.reference.roc_auc,
        )

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run():
        mlflow.set_tag("approved_source_run", cfg.reference.run_id)
        mlflow.set_tag("parity", "passed" if parity_ok else "failed")

        mlflow.log_params(cfg.model_params)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("threshold", cfg.threshold)

        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("pred_time", pred_time)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        with tempfile.TemporaryDirectory() as tmp:
            sweep_path = Path(tmp) / "threshold_sweep.csv"
            sweep.to_csv(sweep_path, index=False)
            mlflow.log_artifact(str(sweep_path))

        signature = infer_signature(X_test, proba)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=cfg.mlflow.registered_model if register else None,
            skops_trusted_types=[
                "churn.features.TelcoFeatureEncoder",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Telco churn XGBoost pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, register=not args.no_register)


if __name__ == "__main__":
    main()
