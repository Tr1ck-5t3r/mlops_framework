from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import yaml
import os

class MLStageConfig(BaseModel):
    enabled: bool = Field(default=True)
    target_column: str = Field(default="actual_delay_days")
    categorical_features: List[str] = Field(default_factory=list)
    numeric_features: List[str] = Field(default_factory=list)
    train_test_split_ratio: float = Field(default=0.8)
    random_seed: int = Field(default=42)
    hyperparameters: Dict[str, Any] = Field(
        default_factory=lambda: {"maxDepth": 5, "maxIter": 20, "stepSize": 0.1}
    )
    # Optional hyperparameter tuning configuration.
    # Example:
    # tuning: {
    #   "enabled": true,
    #   "num_folds": 3,
    #   "metric": "rmse",
    #   "param_grid": {"maxDepth": [3,5], "maxIter": [20,50], "stepSize": [0.05,0.1]}
    # }
    tuning: Optional[Dict[str, Any]] = Field(default=None)

class JoinSpec(BaseModel):
    table: str = Field(..., description="Delta Lake path or table name to join")
    on: List[str] = Field(..., description="Shared join key column name(s)")
    type: str = Field(default="left", description="Join type: inner, left, right, full, cross")


class BronzeTableConfig(BaseModel):
    name: Optional[str] = Field(default=None, description="Logical Bronze table name")
    landing_path: str = Field(..., description="Directory where raw files are landed")
    archive_path: Optional[str] = Field(default=None, description="Directory to archive processed files")
    file_format: str = Field(default="json", description="Landed file format (json, parquet, csv)")
    file_pattern: str = Field(default="*", description="Glob pattern matching batch files")
    target_delta_path: str = Field(..., description="Target Delta Lake path for Bronze layer")
    write_mode: str = Field(default="append", description="Spark save mode (append or overwrite)")
    options: Dict[str, str] = Field(default_factory=dict, description="Spark reader options")


class BronzeConfig(BaseModel):
    landing_path: str = Field(..., description="Directory where external tools land raw files")
    archive_path: Optional[str] = Field(default=None, description="Directory to archive processed files")
    file_format: str = Field(default="json", description="Landed file format (json, parquet, csv)")
    file_pattern: str = Field(default="*", description="Glob pattern matching batch files")
    target_delta_path: str = Field(..., description="Target Delta Lake path for Bronze layer")
    write_mode: str = Field(default="append", description="Spark save mode (append or overwrite)")
    options: Dict[str, str] = Field(default_factory=dict, description="Spark reader options")


class ValidationRule(BaseModel):
    name: str = Field(..., description="Identifier for the validation check")
    expr: str = Field(..., description="PySpark SQL boolean expression")


class SilverConfig(BaseModel):
    source_delta_path: str = Field(...)
    target_delta_path: str = Field(...)
    quarantine_delta_path: Optional[str] = Field(default=None, description="Delta path for quarantined records")
    write_mode: str = Field(default="overwrite")
    drop_duplicates: Optional[List[str]] = Field(default=None)
    drop_na_cols: Optional[List[str]] = Field(default=None)
    fill_na: Optional[Dict[str, Any]] = Field(default=None)
    validation_rules: Optional[List[ValidationRule]] = Field(default=None, description="Quality assertions")


class GoldConfig(BaseModel):
    primary_table: Optional[str] = Field(default=None, description="Root Silver Delta table path")
    source_delta_path: Optional[str] = Field(default=None, description="Legacy Silver Delta path for aggregation")
    target_delta_path: str = Field(..., description="Destination Gold Delta path")
    write_mode: str = Field(default="overwrite", description="Spark write mode")
    joins: Optional[List[JoinSpec]] = Field(default_factory=list, description="Array of dynamic join specs")
    select_expressions: Optional[List[str]] = Field(
        default=None,
        description="Optional SQL expressions/columns to select post-join"
    )
    group_by: Optional[List[str]] = Field(default=None, description="Legacy group-by columns for aggregation")
    aggregations: Optional[Dict[str, str]] = Field(default=None, description="Legacy aggregation definitions")


class MLflowConfig(BaseModel):
    experiment_name: str
    run_name: Optional[str] = Field(default="declarative_run")


class MaintenanceConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable or disable automated maintenance")
    retention_hours: int = Field(default=168, description="Retention threshold in hours for VACUUM (168h = 7 days)")
    tables: Optional[List[str]] = Field(default=None, description="Explicit list of Delta paths to maintain")


class PipelineConfig(BaseModel):
    pipeline_name: str
    version: str
    bronze: Optional[BronzeConfig] = None
    bronze_tables: Optional[List[BronzeTableConfig]] = None
    silver: SilverConfig
    gold: GoldConfig
    maintenance: Optional[MaintenanceConfig] = Field(default_factory=MaintenanceConfig)
    mlflow: MLflowConfig
    ml_stage: Optional[MLStageConfig] = Field(default=None)


def load_pipeline_config(yaml_path: str) -> PipelineConfig:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    return PipelineConfig(**raw_config)