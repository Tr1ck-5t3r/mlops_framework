from mlflow.tracking import MlflowClient
from engine.register_and_promote import MODEL_NAME

def main():
    client = MlflowClient()
    try:
        rm = client.get_registered_model(MODEL_NAME)
        print('Registered model found:', rm.name)
        for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
            print('Version', v.version, 'current_stage', v.current_stage)
        aliases = client.get_registered_model(MODEL_NAME).aliases
        print('Aliases:', aliases)
    except Exception as e:
        print('Model not found in registry or error:', e)

if __name__ == '__main__':
    main()
