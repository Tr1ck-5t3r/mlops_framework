import sys
sys.path.insert(0, r'E:\dbx_mlops')
from engine.config_parser import load_pipeline_config
import mlflow

cfg = load_pipeline_config('config/olist_ml_train.yaml')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
if exp is None:
    print('No experiment found:', cfg.mlflow.experiment_name)
    sys.exit(1)

runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
if not runs:
    print('No runs found in experiment')
    sys.exit(1)

r = runs[0]
print('run_id', r.info.run_id)
print('metrics')
for k, v in r.data.metrics.items():
    print(' ', k, ':', v)
print('params')
for k, v in r.data.params.items():
    print(' ', k, ':', v)

print('artifacts at root:')
for a in client.list_artifacts(r.info.run_id, ''):
    print(' ', a.path)

# list charts folder artifacts if exists
print('\nCharts:')
for a in client.list_artifacts(r.info.run_id, 'charts'):
    print(' ', a.path)
