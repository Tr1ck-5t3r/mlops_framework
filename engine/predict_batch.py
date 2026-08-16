import os
import sys
from typing import Optional

import mlflow
import mlflow.spark
from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def get_spark_session(app_name: str = "BatchInferenceEngine") -> SparkSession:
    """Initializes a Spark Session configured with Delta Lake."""
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", os.path.abspath("data/lakehouse"))
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def get_latest_run_id(experiment_name: str) -> str:
    """Fetches the latest successful MLflow run ID for a given experiment name."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"❌ MLflow experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(f"❌ No finished runs found in experiment '{experiment_name}'.")

    run_id = runs.iloc[0].run_id
    print(f"🔎 Located latest finished MLflow Run ID: {run_id}")
    return run_id


def run_batch_inference(
    experiment_name: str,
    silver_source_path: str,
    output_delta_path: str,
    run_id: Optional[str] = None,
    model_name: str = "GBT_Delivery_Delay_Pipeline",
    model_stage: str = "production",
    id_columns: Optional[list] = None,
) -> DataFrame:
    """Loads the logged MLflow PipelineModel and generates predictions on Silver Delta data.

    Args:
        experiment_name: Name of the MLflow experiment.
        silver_source_path: Delta Lake path to input Silver data.
        output_delta_path: Target Delta Lake path to save predictions.
        run_id: Optional MLflow run ID. If None, resolves the latest production model or latest run ID automatically.
        model_name: Registered MLflow model name.
        model_stage: Registry alias or stage to load from.
        id_columns: Columns to keep in output alongside the predicted value.
    """
    spark = get_spark_session()
    id_cols = id_columns or ["order_id", "customer_id", "vendor_id"]

    # 1. Resolve MLflow Run ID and Model URI
    if run_id:
        model_uri = f"runs:/{run_id}/gbt_delivery_model"
    else:
        model_uri = f"models:/{model_name}@{model_stage}"

    print(f"📥 Loading Spark ML PipelineModel from: {model_uri}")

    try:
        pipeline_model = mlflow.spark.load_model(model_uri)
    except Exception as exc:
        if run_id is None:
            print(f"⚠️ Failed to load registry model '{model_uri}'. Falling back to latest finished run. Error: {exc}")
            run_id = get_latest_run_id(experiment_name)
            model_uri = f"runs:/{run_id}/gbt_delivery_model"
            print(f"📥 Loading Spark ML PipelineModel from fallback URI: {model_uri}")
            pipeline_model = mlflow.spark.load_model(model_uri)
        else:
            raise

    # 3. Read unseen/new data from Silver Delta Lake
    print(f"📂 Reading input batch records from: {silver_source_path}")
    df_silver = spark.read.format("delta").load(silver_source_path)

    # 4. Generate Predictions
    # Note: PipelineModel automatically applies StringIndexerModel -> VectorAssembler -> GBTRegressor
    print("⚙️ Running batch prediction transformation...")
    df_transformed = pipeline_model.transform(df_silver)

    # 5. Format Prediction Payload with Audit Metadata
    df_predictions = (
        df_transformed
        .withColumn("_predicted_at", F.current_timestamp())
        .withColumn("_model_run_id", F.lit(run_id))
        .select(
            *id_cols,
            "predicted_delay_days",
            "_predicted_at",
            "_model_run_id"
        )
    )

    # 6. Write Predictions to Gold Delta Table
    print(f"💾 Saving predictions to Delta Lake: {output_delta_path}")
    df_predictions.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .save(output_delta_path)

    record_count = df_predictions.count()
    print(f"✅ Batch inference complete! Wrote {record_count} predictions to {output_delta_path}")

    return df_predictions


if __name__ == "__main__":
    # Example execution
    run_batch_inference(
        experiment_name="Default",
        silver_source_path="data/lakehouse/silver/orders",
        output_delta_path="data/lakehouse/gold/predictions_delivery_delay"
    )