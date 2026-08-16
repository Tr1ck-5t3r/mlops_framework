from mlflow.tracking import MlflowClient
import mlflow

MODEL_NAME = "GBT_Delivery_Delay_Pipeline"

def register_and_promote_model(run_id: str, target_stage: str = "staging") -> str:
    """Registers a model version from a run ID and sets a lifecycle alias (e.g., 'staging' or 'production')."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/gbt_delivery_model"

    # 1. Register model in MLflow Model Registry
    print(f"📦 Registering model from run '{run_id}' under name '{MODEL_NAME}'...")
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    
    version_num = model_version.version
    print(f"✅ Registered Model Version: {version_num}")

    # 2. Assign Lifecycle Alias ('staging' or 'production')
    # Aliases act as dynamic pointers (e.g., models:/GBT_Delivery_Delay_Pipeline@production)
    alias = target_stage.lower()
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=version_num
    )
    
    print(f"🏷️ Assigned alias '@{alias}' to version {version_num}")
    
    # 3. Add Version Metadata / Description
    client.update_model_version(
        name=MODEL_NAME,
        version=version_num,
        description=f"Promoted to {alias.upper()} on {mlflow.utils.time.get_current_time_millis()}"
    )

    return version_num


def promote_to_production(version: str) -> None:
    """Promotes a specific model version from Staging to Production."""
    client = MlflowClient()
    
    # Reassign the @production alias to the target version
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=version
    )
    print(f"🚀 Model version {version} is now active under @production alias!")


if __name__ == "__main__":
    # Example usage with run ID
    RUN_ID = "YOUR_MLFLOW_RUN_ID"
    
    # Register & set as staging
    v = register_and_promote_model(run_id=RUN_ID, target_stage="staging")
    
    # Promote to production after validation
    promote_to_production(version=v)