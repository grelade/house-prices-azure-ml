from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import load_component
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline


credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")

ml_client = MLClient.from_config(credential=credential)


component_yaml_dir = Path("components")

data_prep = load_component(source = component_yaml_dir / "data_prep.yml")
train_kernelridge = load_component(source = component_yaml_dir / "train_kernelridge.yml")
kaggle_upload = load_component(source = component_yaml_dir / 'kaggle_upload.yml')


@pipeline()
def house_prices_pipeline(data_dir, kernel, submission_message):

    data_prep_result = data_prep(data_dir = data_dir)
    prep_data_dir = data_prep_result.outputs.prep_data_dir

    train_kernelridge_result = train_kernelridge(prep_data_dir = prep_data_dir, kernel = kernel)

    results_dir = train_kernelridge_result.outputs.results_dir
    kaggle_upload_result = kaggle_upload(results_dir = results_dir, submission_message = submission_message)

    return {
        "prep_data_dir": data_prep_result.outputs.prep_data_dir,
        "results_dir": train_kernelridge_result.outputs.results_dir
    }


data_dir = Input(type=AssetTypes.URI_FOLDER, path="azureml:data-dir:1")
kernel = 'linear'
submission_message = 'model message 123'

pipeline_job = house_prices_pipeline(data_dir = data_dir,
                                     kernel = kernel,
                                     submission_message = submission_message)


pipeline_job.outputs.prep_data_dir.mode = 'upload'
pipeline_job.outputs.results_dir.mode = 'upload'

pipeline_job.outputs.prep_data_dir.name = 'prep_data_dir'
pipeline_job.outputs.results_dir.name = 'results_dir'

pipeline_job.settings.default_compute = "aml-cluster"
pipeline_job.settings.default_datastore = "workspaceblobstore"



# submit job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_house_prices"
)
print(pipeline_job)
