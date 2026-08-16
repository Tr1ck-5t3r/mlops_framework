import argparse

import argparse

from engine.config_parser import load_pipeline_config
from engine.runner import MedallionRunner


def main() -> None:
    config_path = "config/sample_pipeline.yaml"
    print(f"📄 Loading configuration from: {config_path}")
    
    config = load_pipeline_config(config_path)
    
    runner = MedallionRunner(config)
    runner.execute_pipeline()

if __name__ == "__main__":
    main()