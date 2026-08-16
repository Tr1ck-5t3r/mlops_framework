"""Wrapper script to run the declarative medallion pipeline for Olist training.

Usage:
    python scripts/train_olist.py --config config/olist_ml_train.yaml

This will execute Bronze -> Silver -> Gold -> ML Training -> Maintenance inside an MLflow run.
"""

import argparse
import sys
from engine.config_parser import load_pipeline_config
from engine.runner import MedallionRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Olist Medallion pipeline and ML training")
    parser.add_argument("--config", "-c", default="config/olist_ml_train.yaml", help="Path to pipeline YAML config")
    args = parser.parse_args(argv)

    cfg = load_pipeline_config(args.config)
    runner = MedallionRunner(cfg)

    try:
        runner.execute_pipeline()
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
