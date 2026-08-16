# How to run the Declarative Medallion Pipeline and Model Registration

This document summarizes the common run commands, quick verification steps, and troubleshooting tips for the medallion pipeline, ML training, model registration, and batch inference.

Prerequisites
- Activate the project's Python virtualenv (example):

```powershell
& 'E:\dbx_mlops\venv\Scripts\Activate.ps1'
pip install -r requirements.txt
```

1) Run the full pipeline (Bronze -> Silver -> Gold -> ML training)

```powershell
python scripts/train_olist.py --config config/olist_ml_train.yaml
```

Notes:
- The pipeline writes Delta tables under paths configured in the YAML.
- MLflow logs are written to the local `mlruns/` directory by default.

2) Register the latest model and run batch inference

This script registers the most recent MLflow run's model, writes a compact JSON
registration log, and then runs batch inference on the Gold features.

```powershell
python scripts/register_and_run_inference.py
```

Important artifacts:
- Compact registration log: `scripts/register_compact_log.json`
  - Quick-check this file immediately after running the script to confirm
    whether model registration succeeded before any Spark logs appear.
  - Example content:

```json
{
  "run_id": "984437d473fb40fbb250cf5b785ecbec",
  "model_name": "GBT_Delivery_Delay_Pipeline",
  "success": true,
  "version": "1",
  "error": null,
  "stage": "staging",
  "timestamp": 1692100000
}
```

- Predictions output (if batch inference succeeds):
  - Parquet folder: `data/predictions/predictions_<run_id>`
  - Contents: Gold feature columns + `predicted_delay_days`.

3) Inspect the registry (optional)

```powershell
python scripts/check_registry.py
```

4) Register-only (no Spark):

```powershell
python scripts/register_only.py
```

This registers the latest run into the model registry and writes the registration
result via stdout.

5) Troubleshooting
- If Spark prints many heartbeater / BlockManager logs and floods the console:
  - Check `scripts/register_compact_log.json` for registration status — the log
    is written before Spark starts to make verification reliable.
  - The inference script attempts to set Spark and JVM logging to ERROR to
    reduce noise. If logs still flood, consider running on a machine with fewer
    background Spark processes or increasing log suppression in `log4j.properties`.

- If `success` is false in the compact log:
  - Read the `error` string to determine cause (auth, registry connectivity,
    duplicate model name, etc.).
  - Use `python scripts/check_registry.py` to query the registry and confirm
    state.

6) Notes on environment
- MLflow UI (optional): run `mlflow ui` from project root and open http://localhost:5000
- If using a remote MLflow server, ensure `MLFLOW_TRACKING_URI` is set appropriately

If you want, I can also add a tiny Makefile or PS1 helper script to wrap these commands for convenience.