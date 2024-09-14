from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")

ml_client = MLClient.from_config(credential=credential)

custom_env_name = "kaggle-env"

kaggle_upload_env = Environment(
    name=custom_env_name,
    description="Custom env for kaggle submission",
    conda_file="dependencies/conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.1"
)

ml_client.environments.create_or_update(kaggle_upload_env)

print(
    f"Environment with name {kaggle_upload_env.name} is registered to workspace, the environment version is {kaggle_upload_env.version}"
)
