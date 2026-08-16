# Olist ML Training (Quick Start)

This document shows how to run the declarative medallion pipeline and train the delivery delay model on the Olist dataset.

Prerequisites:
- Java + Spark available on PATH (local mode)
- Python 3.10+ with dependencies from `requirements.txt` installed
- Optional: `MLFLOW_TRACKING_URI` environment variable if you want a remote MLflow server

Run the full pipeline (Bronze -> Silver -> Gold -> ML Training):

```bash
python scripts/train_olist.py --config config/olist_ml_train.yaml
```

Open the notebook report (after running the pipeline):

```bash
# From project root
jupyter lab notebooks/olist_ml_report.ipynb
```

Notes:
- The pipeline writes Delta tables under `E:/dbx_mlops/data/delta/...` per `config/olist_ml_train.yaml`.
- MLflow logs are stored in `mlruns/` (local) unless `MLFLOW_TRACKING_URI` is set.
