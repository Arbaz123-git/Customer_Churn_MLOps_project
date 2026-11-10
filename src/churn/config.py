from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int
    stratify: bool


@dataclass(frozen=True)
class FeatureConfig:
    binary: list[str]
    categorical: list[str]


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment: str
    registered_model: str


@dataclass(frozen=True)
class ReferenceConfig:
    run_id: str
    recall: float
    roc_auc: float
    tolerance: float


@dataclass(frozen=True)
class Config:
    raw_path: Path
    id_col: str
    target: str
    features: FeatureConfig
    split: SplitConfig
    threshold: float
    model_params: dict[str, Any]
    mlflow: MlflowConfig
    reference: ReferenceConfig


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    raw_path = Path(raw["data"]["raw_path"])
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    tracking_uri = raw["mlflow"]["tracking_uri"]
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_path = tracking_uri.removeprefix("sqlite:///")
        if not Path(db_path).is_absolute():
            tracking_uri = f"sqlite:///{PROJECT_ROOT / db_path}"

    return Config(
        raw_path=raw_path,
        id_col=raw["data"]["id_col"],
        target=raw["data"]["target"],
        features=FeatureConfig(
            binary=list(raw["features"]["binary"]),
            categorical=list(raw["features"]["categorical"]),
        ),
        split=SplitConfig(**raw["split"]),
        threshold=float(raw["threshold"]),
        model_params=dict(raw["model"]["params"]),
        mlflow=MlflowConfig(
            tracking_uri=tracking_uri,
            experiment=raw["mlflow"]["experiment"],
            registered_model=raw["mlflow"]["registered_model"],
        ),
        reference=ReferenceConfig(**raw["reference"]),
    )
