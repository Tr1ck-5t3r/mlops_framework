import sys
sys.path.insert(0, r'E:\dbx_mlops')
from engine.config_parser import load_pipeline_config
import mlflow
import json

cfg = load_pipeline_config('config/olist_ml_train.yaml')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
out = {'experiment': cfg.mlflow.experiment_name}
if exp is None:
    out['error'] = 'no experiment'
else:
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        out['error'] = 'no runs'
    else:
        r = runs[0]
        out['run_id'] = r.info.run_id
        out['metrics'] = r.data.metrics
        out['params'] = r.data.params
        # try to read model metadata file for GBT params if exists
        try:
            art_path = client.list_artifacts(r.info.run_id, 'gbt_delivery_model')
            out['artifacts'] = [a.path for a in art_path]
        except Exception:
            out['artifacts'] = []

with open('scripts/report_result.txt', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)

print('WROTE scripts/report_result.txt')
