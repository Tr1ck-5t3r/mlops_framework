import os
import json
import time
from mlflow.tracking import MlflowClient
import mlflow
from engine.config_parser import load_pipeline_config
from engine.register_and_promote import register_and_promote_model, MODEL_NAME
from engine.runner import get_spark_session


def main(config_path: str = 'config/olist_ml_train.yaml', target_stage: str = 'staging'):
    cfg = load_pipeline_config(config_path)
    client = MlflowClient()

    exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {cfg.mlflow.experiment_name}")

    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError('No runs found in experiment')

    run = runs[0]
    run_id = run.info.run_id
    print('Latest run id:', run_id)

    # Write an initial compact log immediately so a file exists before any heavy operations
    try:
        initial = {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "stage": target_stage,
            "status": "started",
            "timestamp": int(time.time())
        }
        os.makedirs('scripts', exist_ok=True)
        with open(os.path.join('scripts', 'register_compact_log.json'), 'w', encoding='utf-8') as f:
            json.dump(initial, f)
        print('WROTE initial scripts/register_compact_log.json')
    except Exception as e:
        print('Failed to write initial compact log:', e)

    # Register and promote (capture failures but always write a compact log BEFORE starting Spark)
    reg_success = False
    version = None
    reg_error = None
    try:
        version = register_and_promote_model(run_id=run_id, target_stage=target_stage)
        reg_success = True
        print(f'Registered model {MODEL_NAME} version {version} under @{target_stage}')
    except Exception as e:
        reg_error = str(e)
        print('Model registration failed:', reg_error)

    # Write a compact registration log BEFORE starting Spark (helps verify registration even if Spark logs are noisy)
    try:
        log = {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "success": bool(reg_success),
            "version": str(version) if version is not None else None,
            "error": reg_error,
            "stage": target_stage,
            "timestamp": int(time.time())
        }
        os.makedirs('scripts', exist_ok=True)
        with open(os.path.join('scripts', 'register_compact_log.json'), 'w', encoding='utf-8') as f:
            json.dump(log, f)
        print('WROTE scripts/register_compact_log.json')
    except Exception as e:
        print('Failed to write compact registration log:', e)

    # Load model from registry alias (models:/NAME@stage)
    model_uri = f"models:/{MODEL_NAME}@{target_stage}"
    print('Loading model from registry URI:', model_uri)

    # Start Spark and load Gold features
    spark = get_spark_session('RegisterAndInference')
    # Quiet Spark and JVM logging to reduce noisy heartbeater logs in local runs
    try:
        import logging
        logging.getLogger('py4j').setLevel(logging.ERROR)
        spark.sparkContext.setLogLevel('ERROR')
        # also set the JVM log4j root logger level to ERROR when available
        try:
            log4j = spark._jvm.org.apache.log4j
            log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
        except Exception:
            pass
    except Exception:
        pass
    gold_path = cfg.gold.target_delta_path
    print('Reading Gold features from:', gold_path)
    df_gold = spark.read.format('delta').load(gold_path)

    # Load model and run batch inference
    try:
        pipeline_model = mlflow.spark.load_model(model_uri)
    except Exception as e:
        print('Could not load model from registry alias, falling back to run artifact:', e)
        pipeline_model = mlflow.spark.load_model(f'runs:/{run_id}/gbt_delivery_model')

    print('Running batch inference on Gold table...')
    preds = pipeline_model.transform(df_gold)

    # Normalize prediction column name
    pred_col = None
    for c in preds.columns:
        if c.lower() == 'predicted_delay_days' or c.lower() == 'prediction' or c.lower().endswith('prediction'):
            pred_col = c
            break
    if pred_col is None:
        # try default 'prediction'
        pred_col = 'prediction' if 'prediction' in preds.columns else None

    if pred_col and pred_col != 'predicted_delay_days':
        preds = preds.withColumnRenamed(pred_col, 'predicted_delay_days')

    out_dir = os.path.join('data', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'predictions_{run_id}')

    # Write as parquet for easy consumption
    print('Writing predictions to:', out_path)
    preds.write.mode('overwrite').parquet(out_path)

    print('Batch inference complete. Predictions stored at:', out_path)
    print('MLflow run:', run_id)
    print('Registered version:', version)


if __name__ == '__main__':
    main()
