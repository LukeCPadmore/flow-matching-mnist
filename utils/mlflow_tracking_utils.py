from mlflow.tracking import MlflowClient

def get_run_param(run_id: str, key: str) -> str:
    client = MlflowClient()
    run = client.get_run(run_id)
    if key not in run.data.params:
        raise KeyError(f"Run {run_id} has no param '{key}'. Available: {list(run.data.params)[:20]} ...")
    return run.data.params[key] 