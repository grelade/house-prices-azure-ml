import pandas as pd
import os

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")

ml_client = MLClient.from_config(credential=credential)



train_data = Data(
    path=f'./data/train.csv',
    type=AssetTypes.URI_FILE,
    description="Training data asset",
    name="training-data"
)

test_data = Data(
    path=f'./data/test.csv',
    type=AssetTypes.URI_FILE,
    description="Test data asset",
    name="test-data"
)

data_description = Data(
    path=f'./data/data_description.txt',
    type=AssetTypes.URI_FILE,
    description="data description file",
    name="data-desc"
)

data_directory = Data(path = f'./data/', 
type = AssetTypes.URI_FOLDER, 
description="data directory",
name='data-dir')

ml_client.data.create_or_update(train_data)
ml_client.data.create_or_update(test_data)
ml_client.data.create_or_update(data_description)
ml_client.data.create_or_update(data_directory)

data = ml_client.data.list()
for datum in data:
    print(datum.name)


data_asset = ml_client.data.get("training-data",version=1)
print(pd.read_csv(data_asset.path).head())

data_asset = ml_client.data.get("test-data", version=1)
print(pd.read_csv(data_asset.path).head())

data_asset = ml_client.data.get("data-desc", version=1)

# txt file must be downloaded to be read (???)
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
artifact_utils.download_artifact_from_aml_uri(data_asset.path, 
destination ='.', 
datastore_operation=ml_client.datastores)

with open('./data_description.txt','r') as fp:
    
    for line in list(fp.readlines())[:5]:
        print(line)

os.remove('./data_description.txt')

