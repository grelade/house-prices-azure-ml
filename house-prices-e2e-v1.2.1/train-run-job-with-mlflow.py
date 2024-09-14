from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import command
from azure.ai.ml.sweep import Choice


credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")

ml_client = MLClient.from_config(credential=credential)


job_inputs = {'train_data': Input(type=AssetTypes.URI_FILE, path="azureml:training-data:1"), 
'test_data': Input(type=AssetTypes.URI_FILE, path="azureml:test-data:1"), 
'data_description': Input(type=AssetTypes.URI_FILE, path="azureml:data-desc:1"), 
'kernel': 'linear'}

cmd = "python aml_train_model.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} --data_description ${{inputs.data_description}} --kernel ${{inputs.kernel}}"
# configure job
job = command(
    code = "./src",
    command = cmd,
    inputs = job_inputs,
    environment = "AzureML-sklearn-1.5@latest",
    compute = "aml-cluster",
    display_name = "house-prices-e2e-v2-2",
    experiment_name = "house-prices-e2e-v2-2-training"
)
                 
command_job_for_sweep = job(
    kernel=Choice(values=['linear', 'rbf', 'poly']))

# apply the sweep parameter to obtain the sweep_job
sweep_job = command_job_for_sweep.sweep(
    compute="aml-cluster",
    sampling_algorithm="grid",
    primary_metric="mse_test",
    goal="Minimize",
)

# set the name of the sweep job experiment
sweep_job.experiment_name="sweep-kernel"

# define the limits for this sweep
sweep_job.set_limits(max_total_trials=2, max_concurrent_trials=1, timeout=7200)

returned_sweep_job = ml_client.create_or_update(sweep_job)
aml_url = returned_sweep_job.studio_url
print("Monitor your job at", aml_url)
