import pandas as pd
import os

import mltable
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

credential = DefaultAzureCredential()
credential.get_token("https://management.azure.com/.default")

ml_client = MLClient.from_config(credential=credential)

data_directory = Data(path = f'./data/',
type = AssetTypes.URI_FOLDER,
description="data directory",
name='data-dir')

ml_client.data.create_or_update(data_directory)

data = ml_client.data.list()
for datum in data:
    print(datum.name)


# access csv files using mltable
data_asset = ml_client.data.get("data-dir",version=1)
data_dir = data_asset.path

df = mltable.from_delimited_files(paths=[{'file': data_dir + 'train.csv'}]).to_pandas_dataframe()
print(df.head())

df = mltable.from_delimited_files(paths=[{'file': data_dir + 'test.csv'}]).to_pandas_dataframe()
print(df.head())

# access csv files using Input class
data_dir_input = Input(type = AssetTypes.URI_FOLDER, path = data_dir)
print(pd.read_csv(data_dir_input.path + 'train.csv').head())
print(pd.read_csv(data_dir_input.path + 'test.csv').head())


# txt file must be downloaded to be read (???)
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
artifact_utils.download_artifact_from_aml_uri(data_dir + 'data_description.txt',
destination ='.',
datastore_operation=ml_client.datastores)

with open('./data_description.txt','r') as fp:

    for line in list(fp.readlines())[:5]:
        print(line)

os.remove('./data_description.txt')
