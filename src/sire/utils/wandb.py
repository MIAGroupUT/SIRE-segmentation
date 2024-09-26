import wandb


def load_wandb_artifact(project_name: str, run_name: str, alias: str, **kwargs):
    # Retrieve artifact
    api = wandb.Api()
    artifact_run = [run.id for run in api.runs(project_name) if run.name == run_name][0]
    artifact = api.artifact(f"{project_name}/model-{artifact_run}:{alias}")

    # Download artifact
    run = wandb.init(project="download")
    artifact_dir = run.use_artifact(artifact).download()

    run.finish()
    # api.run(f"donwload/{run.id}").delete()

    return artifact_dir
