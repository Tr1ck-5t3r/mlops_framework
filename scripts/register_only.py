from mlflow.tracking import MlflowClient
from engine.config_parser import load_pipeline_config
from engine.register_and_promote import register_and_promote_model, MODEL_NAME

def main(config_path: str = 'config/olist_ml_train.yaml'):
    cfg = load_pipeline_config(config_path)
    client = MlflowClient()
    exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {cfg.mlflow.experiment_name}")
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError('No runs found')
    run = runs[0]
    run_id = run.info.run_id
    print('Latest run id:', run_id)
    version = register_and_promote_model(run_id=run_id, target_stage='staging')
    print(f'Registered {MODEL_NAME} as version {version} @staging')

if __name__ == '__main__':
    main()
