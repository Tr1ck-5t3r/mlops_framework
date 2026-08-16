import glob
import os
import shutil
import sys
import tempfile
from datetime import datetime
from functools import reduce
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import mlflow
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from engine.config_parser import PipelineConfig


def get_spark_session(app_name: str = "DeclarativeMedallionEngine") -> SparkSession:
    """Initializes Spark Session with Delta Lake and local environment compatibility settings."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hadoop_dir = os.path.join(project_root, "hadoop")

    if os.path.exists(hadoop_dir):
        os.environ["HADOOP_HOME"] = hadoop_dir
        os.environ["PATH"] = os.path.join(hadoop_dir, "bin") + os.pathsep + os.environ.get("PATH", "")

    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    tmp_dir = os.path.join(tempfile.gettempdir(), "spark_local_dir")

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", os.path.abspath("data/lakehouse"))
        .config("spark.local.dir", tmp_dir)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


class MedallionRunner:
    """Executes Bronze, Silver, Gold, ML Training, and Maintenance tasks dynamically based on PipelineConfig."""

    def __init__(self, config: PipelineConfig, spark: SparkSession = None):
        self.config = config
        self.spark = spark or get_spark_session(config.pipeline_name)

    def _archive_processed_files(self, file_list: List[str], archive_path: str) -> None:
        """Moves processed landing files to the archive directory with a timestamp prefix."""
        os.makedirs(archive_path, exist_ok=True)
        timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

        archived_count = 0
        for file_path in file_list:
            if os.path.isfile(file_path):
                filename = os.path.basename(file_path)
                archived_filename = f"{timestamp_prefix}_{filename}"
                dest_path = os.path.join(archive_path, archived_filename)

                shutil.move(file_path, dest_path)
                archived_count += 1

        print(f"📦 Archived {archived_count} file(s) to: {archive_path}")

    def _run_single_bronze(self, cfg: Any) -> int:
        landing_glob = os.path.join(cfg.landing_path, cfg.file_pattern)
        matched_files = glob.glob(landing_glob)

        if not matched_files:
            print(f"⚠️ No landed files found matching pattern: {landing_glob}")
            return 0

        print(f"📂 Found {len(matched_files)} file(s) in landing path matching: {cfg.file_pattern}")

        reader = self.spark.read.format(cfg.file_format)
        if cfg.options:
            reader = reader.options(**cfg.options)

        df_landing = reader.load(matched_files)

        df_bronze = (
            df_landing
            .withColumn("_ingested_at", F.current_timestamp())
            .withColumn("_source_file", F.input_file_name())
        )

        writer = df_bronze.write.format("delta").mode(cfg.write_mode)
        if cfg.write_mode == "append":
            writer = writer.option("mergeSchema", "true")

        writer.save(cfg.target_delta_path)

        count = df_bronze.count()
        print(f"✅ Bronze write complete for target '{cfg.target_delta_path}'. Rows written: {count}")

        if getattr(cfg, "archive_path", None):
            self._archive_processed_files(matched_files, cfg.archive_path)

        return count

    def run_bronze(self) -> int:
        """Bronze Stage: Ingest raw landed files into Delta Lake and archive them."""
        print("\n=== 🥉 Executing Bronze Stage (Landing Sweep & Archive) ===")

        total_count = 0
        if self.config.bronze_tables:
            for table_cfg in self.config.bronze_tables:
                name = getattr(table_cfg, "name", None) or os.path.basename(table_cfg.target_delta_path)
                print(f"🔹 Processing Bronze table: {name}")
                total_count += self._run_single_bronze(table_cfg)
        elif self.config.bronze:
            total_count = self._run_single_bronze(self.config.bronze)
        else:
            print("⚠️ No Bronze ingestion configuration provided. Skipping Bronze stage.")

        return total_count

    def run_silver(self) -> Dict[str, Any]:
        """Silver Stage: Cleanse, validate assertions, route bad records to quarantine, and return DQ metrics."""
        print("\n=== 🥈 Executing Silver Stage ===")
        cfg = self.config.silver

        # Read from Bronze Delta Lake
        df = self.spark.read.format("delta").load(cfg.source_delta_path)

        # 1. Impute missing values
        if cfg.fill_na:
            df = df.fillna(cfg.fill_na)

        # 2. Drop duplicates
        if cfg.drop_duplicates:
            df = df.dropDuplicates(cfg.drop_duplicates)

        # 3. Drop NA rows for specific columns
        if cfg.drop_na_cols:
            df = df.dropna(subset=cfg.drop_na_cols)

        quarantine_count = 0

        # 4. Evaluate Validation Rules & Quarantine Invalid Data
        if cfg.validation_rules and cfg.quarantine_delta_path:
            print("🛡️ Evaluating Data Quality Assertions...")
            rule_conds = []
            failed_rule_exprs = []

            for rule in cfg.validation_rules:
                cond = F.expr(rule.expr)
                rule_conds.append(cond)

                # Evaluates condition; tags rule name if condition fails
                failed_expr = F.when(~cond | cond.isNull(), F.lit(rule.name)).otherwise(F.lit(None))
                failed_rule_exprs.append(failed_expr)

            if rule_conds:
                # Combine all conditions safely using reduce
                all_valid_cond = reduce(lambda a, b: a & b, rule_conds)

                df_evaluated = df.withColumn(
                    "_failed_rules",
                    F.array_remove(F.array(*failed_rule_exprs), None)
                )

                df_valid = df_evaluated.filter(all_valid_cond).drop("_failed_rules")
                df_quarantine = (
                    df_evaluated
                    .filter(~all_valid_cond | all_valid_cond.isNull())
                    .withColumn("_quarantined_at", F.current_timestamp())
                )

                quarantine_count = df_quarantine.count()
                if quarantine_count > 0:
                    df_quarantine.write.format("delta").mode("append").save(cfg.quarantine_delta_path)
                    print(f"⚠️ Quarantined {quarantine_count} invalid record(s) -> {cfg.quarantine_delta_path}")
                else:
                    print("✨ 100% Data Quality Pass Rate! No records quarantined.")

                df = df_valid

        # Write clean data to Silver Delta Lake
        df.write.format("delta").mode(cfg.write_mode).save(cfg.target_delta_path)
        clean_count = df.count()

        # Compute DQ Metrics
        total_records = clean_count + quarantine_count
        pass_rate_pct = round((clean_count / total_records * 100.0), 2) if total_records > 0 else 100.0
        quarantine_ratio_pct = round((quarantine_count / total_records * 100.0), 2) if total_records > 0 else 0.0

        print(f"✅ Silver cleansing complete. Clean: {clean_count} | Quarantined: {quarantine_count} | Pass Rate: {pass_rate_pct}%")

        return {
            "clean_count": clean_count,
            "quarantine_count": quarantine_count,
            "total_evaluated": total_records,
            "pass_rate_pct": pass_rate_pct,
            "quarantine_ratio_pct": quarantine_ratio_pct,
        }

    def run_gold(self) -> DataFrame:
        """Builds the Gold feature table using joins or legacy aggregation mode."""
        print("\n=== 🏆 Executing Gold Stage (Feature Engineering) ===")
        gold_cfg = self.config.gold

        source_table = gold_cfg.primary_table or gold_cfg.source_delta_path
        if not source_table:
            raise ValueError("Gold stage requires either 'primary_table' or 'source_delta_path'.")

        print(f"📥 Loading Gold source table: {source_table}")
        df = self.spark.read.format("delta").load(source_table)

        if gold_cfg.joins:
            for idx, join_spec in enumerate(gold_cfg.joins, start=1):
                print(f"🔗 [{idx}/{len(gold_cfg.joins)}] Joining '{join_spec.table}'")
                print(f"   ├─ Keys: {join_spec.on} | Type: {join_spec.type.upper()}")

                right_df = self.spark.read.format("delta").load(join_spec.table)
                df = df.join(right_df, on=join_spec.on, how=join_spec.type)

        if gold_cfg.select_expressions:
            print("🎯 Applying post-join column select expressions...")
            df = df.selectExpr(*gold_cfg.select_expressions)
        elif gold_cfg.group_by and gold_cfg.aggregations:
            print("🎯 Applying legacy group-by aggregation mode...")
            agg_exprs = [F.expr(f"{func}({col}) as {col}_{func}") for col, func in gold_cfg.aggregations.items()]
            df = df.groupBy(*gold_cfg.group_by).agg(*agg_exprs)

        print(f"💾 Writing Gold Feature Table to: {gold_cfg.target_delta_path}")
        writer = df.write.format("delta").mode(gold_cfg.write_mode)
        if gold_cfg.write_mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        writer.save(gold_cfg.target_delta_path)

        row_count = df.count()
        print(f"✅ Gold Stage Complete! Total Records: {row_count}")
        return df

    def run_ml_training(self) -> Dict[str, float]:
        """ML Stage: Preprocesses features, trains GBTRegressor, and logs to MLflow."""
        ml_cfg = getattr(self.config, "ml_stage", None)
        if not ml_cfg or not ml_cfg.enabled:
            print("⏩ ML Training Stage is disabled in config. Skipping...")
            return {}

        print("\n=== 🤖 Executing ML Stage: GBTRegressor Delivery Delay Model ===")

        # 1. Read Gold Delta Feature Table
        gold_path = self.config.gold.target_delta_path
        print(f"📥 Loading Gold Features from: {gold_path}")
        df_gold = self.spark.read.format("delta").load(gold_path)

        # 2. Filter out Nulls in Target & Features
        required_cols = [ml_cfg.target_column] + ml_cfg.categorical_features + ml_cfg.numeric_features
        df_ml = df_gold.dropna(subset=required_cols)

        # 3. Build PySpark ML Pipeline Stages
        pipeline_stages = []
        indexed_cat_cols = []

        # A. Index Categorical Columns (String -> Numeric Index)
        for cat_col in ml_cfg.categorical_features:
            indexed_col = f"{cat_col}_indexed"
            indexer = StringIndexer(
                inputCol=cat_col,
                outputCol=indexed_col,
                handleInvalid="keep"
            )
            pipeline_stages.append(indexer)
            indexed_cat_cols.append(indexed_col)

        # B. Assemble Features into a single Dense Vector
        assembler_inputs = indexed_cat_cols + ml_cfg.numeric_features
        assembler = VectorAssembler(
            inputCols=assembler_inputs,
            outputCol="features",
            handleInvalid="skip"
        )
        pipeline_stages.append(assembler)

        # C. Configure GBTRegressor Estimator
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol=ml_cfg.target_column,
            predictionCol="predicted_delay_days",
            maxDepth=ml_cfg.hyperparameters.get("maxDepth", 5),
            maxIter=ml_cfg.hyperparameters.get("maxIter", 20),
            stepSize=ml_cfg.hyperparameters.get("stepSize", 0.1),
            seed=ml_cfg.random_seed
        )
        pipeline_stages.append(gbt)

        # 4. Construct Pipeline & Train/Test Split
        ml_pipeline = Pipeline(stages=pipeline_stages)

        train_ratio = ml_cfg.train_test_split_ratio
        test_ratio = 1.0 - train_ratio
        train_df, test_df = df_ml.randomSplit([train_ratio, test_ratio], seed=ml_cfg.random_seed)

        print(f"📊 Dataset Split -> Train Records: {train_df.count()} | Test Records: {test_df.count()}")

        # 5. Fit Pipeline Model & Predict (optionally with hyperparameter tuning)
        tuning_cfg = getattr(ml_cfg, "tuning", None) or {}
        if tuning_cfg.get("enabled"):
            print("🔎 Hyperparameter tuning enabled — running CrossValidator grid search...")
            param_grid_cfg = tuning_cfg.get("param_grid", {})

            param_builder = ParamGridBuilder()
            for param_name, values in param_grid_cfg.items():
                try:
                    param_obj = getattr(gbt, param_name)
                    param_builder.addGrid(param_obj, values)
                except Exception as e:
                    print(f"⚠️ Skipping tuning param '{param_name}': {e}")

            grid = param_builder.build()
            metric_name = tuning_cfg.get("metric", "rmse")
            evaluator = RegressionEvaluator(labelCol=ml_cfg.target_column, predictionCol="predicted_delay_days", metricName=metric_name)

            cv = CrossValidator(
                estimator=ml_pipeline,
                estimatorParamMaps=grid,
                evaluator=evaluator,
                numFolds=int(tuning_cfg.get("num_folds", 3)),
            )

            print("⚙️ Running cross-validation. This may take a while...")
            cv_model = cv.fit(train_df)
            pipeline_model = cv_model.bestModel
            predictions = pipeline_model.transform(test_df)

            # Try to extract best params from the bestModel's gbt stage
            best_params = {}
            try:
                gbt_best = next((s for s in pipeline_model.stages if hasattr(s, "featureImportances")), None)
                if gbt_best is not None and param_grid_cfg:
                    for pname in param_grid_cfg.keys():
                        try:
                            val = gbt_best.getOrDefault(getattr(gbt_best, pname))
                            best_params[pname] = val
                        except Exception:
                            pass
            except Exception:
                pass

            if mlflow.active_run():
                for k, v in best_params.items():
                    mlflow.log_param(f"best_{k}", v)
                print(f"✅ Hyperparameter tuning complete. Best params: {best_params}")
        else:
            print("⚙️ Training PySpark MLlib GBTRegressor Model...")
            pipeline_model = ml_pipeline.fit(train_df)
            predictions = pipeline_model.transform(test_df)

        # 6. Evaluate Model Metrics
        evaluator_rmse = RegressionEvaluator(labelCol=ml_cfg.target_column, predictionCol="predicted_delay_days", metricName="rmse")
        evaluator_mae = RegressionEvaluator(labelCol=ml_cfg.target_column, predictionCol="predicted_delay_days", metricName="mae")
        evaluator_r2 = RegressionEvaluator(labelCol=ml_cfg.target_column, predictionCol="predicted_delay_days", metricName="r2")

        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        print("\n📈 Model Evaluation Metrics:")
        print(f"   ├─ Root Mean Squared Error (RMSE) : {rmse:.4f} days")
        print(f"   ├─ Mean Absolute Error (MAE)     : {mae:.4f} days")
        print(f"   └─ R-Squared Score (R²)          : {r2:.4f}")

        feature_importance_scores = {}
        gbt_model = next((stage for stage in pipeline_model.stages if hasattr(stage, "featureImportances")), None)
        if gbt_model is not None:
            importances = gbt_model.featureImportances
            feature_importance_scores = {
                feature_name: float(importances[idx])
                for idx, feature_name in enumerate(assembler_inputs)
                if idx < len(importances)
            }

        # 7. Log Metrics, Parameters, and PipelineModel Artifact to MLflow
        if mlflow.active_run():
            mlflow.log_params(ml_cfg.hyperparameters)
            mlflow.log_param("train_test_ratio", train_ratio)
            mlflow.log_param("categorical_features", ml_cfg.categorical_features)
            mlflow.log_param("numeric_features", ml_cfg.numeric_features)

            mlflow.log_metric("test_rmse", round(rmse, 4))
            mlflow.log_metric("test_mae", round(mae, 4))
            mlflow.log_metric("test_r2", round(r2, 4))

            if feature_importance_scores:
                for name, score in feature_importance_scores.items():
                    mlflow.log_metric(f"feature_importance_{name}", round(score, 6))

                fig, ax = plt.subplots(figsize=(10, max(4, len(feature_importance_scores) * 0.5)))
                sorted_importance = sorted(feature_importance_scores.items(), key=lambda item: item[1], reverse=True)
                features, scores = zip(*sorted_importance)
                ax.barh(list(features)[::-1], list(scores)[::-1], color="tab:blue")
                ax.set_xlabel("Importance")
                ax.set_title("GBT Delivery Delay Feature Importance")
                fig.tight_layout()

                chart_path = os.path.join(tempfile.gettempdir(), f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                fig.savefig(chart_path)
                plt.close(fig)
                mlflow.log_artifact(chart_path, artifact_path="charts")
                print(f"💾 Feature importance chart logged to MLflow charts/ folder.")

            mlflow.spark.log_model(pipeline_model, artifact_path="gbt_delivery_model")
            print("💾 PipelineModel and metrics logged successfully to MLflow!")

        return {"rmse": rmse, "mae": mae, "r2": r2}

    def run_maintenance(self) -> Dict[str, str]:
        """Maintenance Stage: Compacts small files (OPTIMIZE) and removes stale files (VACUUM)."""
        print("\n=== 🧹 Executing Maintenance Stage (OPTIMIZE & VACUUM) ===")
        cfg = getattr(self.config, "maintenance", None)

        if not cfg or not cfg.enabled:
            print("⏭️ Maintenance disabled in configuration. Skipping.")
            return {"status": "SKIPPED"}

        target_tables = cfg.tables or [
            self.config.silver.target_delta_path,
            self.config.gold.target_delta_path
        ]

        if cfg.retention_hours == 0:
            self.spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")

        maintenance_results = {}

        for table_path in target_tables:
            if not DeltaTable.isDeltaTable(self.spark, table_path):
                print(f"⚠️ Delta table does not exist or is invalid, skipping maintenance: {table_path}")
                continue

            print(f"🔧 Processing Delta Table: {table_path}")
            try:
                delta_table = DeltaTable.forPath(self.spark, table_path)

                # 1. OPTIMIZE: Bin-pack small Parquet files
                print("   • Running OPTIMIZE file compaction...")
                delta_table.optimize().executeCompaction()

                # 2. VACUUM: Remove stale files older than retention threshold
                print(f"   • Running VACUUM (Retention: {cfg.retention_hours} hours)...")
                delta_table.vacuum(cfg.retention_hours)

                print(f"   ✅ Maintenance complete for: {table_path}")
                maintenance_results[table_path] = "SUCCESS"

            except Exception as e:
                print(f"   ❌ Maintenance failed for {table_path}: {str(e)}")
                maintenance_results[table_path] = f"FAILED: {str(e)}"

        return maintenance_results

    def execute_pipeline(self) -> None:
        """Runs the entire Medallion sequence + ML Training + Maintenance inside an MLflow run."""
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        landing_paths = []
        if self.config.bronze:
            landing_paths.append(self.config.bronze.landing_path)
        if self.config.bronze_tables:
            landing_paths.extend([table.landing_path for table in self.config.bronze_tables])
        landing_path_param = ";".join(sorted(set(landing_paths))) if landing_paths else ""

        with mlflow.start_run(run_name=self.config.mlflow.run_name):
            # Log Parameters
            mlflow.log_param("pipeline_name", self.config.pipeline_name)
            mlflow.log_param("version", self.config.version)
            if landing_path_param:
                mlflow.log_param("landing_path", landing_path_param)

            # Run Core Stages
            bronze_count = self.run_bronze()
            silver_metrics = self.run_silver()
            gold_df = self.run_gold()
            gold_count = gold_df.count()

            # Run ML Training Stage
            ml_metrics = self.run_ml_training()

            # Run Maintenance Stage
            maintenance_results = self.run_maintenance()

            # Log Record Volume Metrics
            mlflow.log_metric("bronze_record_count", bronze_count)
            mlflow.log_metric("silver_record_count", silver_metrics["clean_count"])
            mlflow.log_metric("quarantine_record_count", silver_metrics["quarantine_count"])
            mlflow.log_metric("gold_record_count", gold_count)

            # Log DQ Metrics
            mlflow.log_metric("dq_pass_rate_pct", silver_metrics["pass_rate_pct"])
            mlflow.log_metric("dq_quarantine_ratio_pct", silver_metrics["quarantine_ratio_pct"])

            print(f"\n🎉 Pipeline '{self.config.pipeline_name}' completed successfully!")
            print(f"📈 MLflow Observability Summary:")
            print(f"   • Pass Rate: {silver_metrics['pass_rate_pct']}%")
            print(f"   • Quarantined Records: {silver_metrics['quarantine_count']}")
            print(f"   • Clean Silver Records: {silver_metrics['clean_count']}")
            print(f"   • Gold Feature Records: {gold_count}")
            if ml_metrics:
                print(f"   • ML Test RMSE: {ml_metrics.get('rmse', 0.0):.4f} days")
            print(f"   • Maintenance Executed: {len(maintenance_results)} table(s)")