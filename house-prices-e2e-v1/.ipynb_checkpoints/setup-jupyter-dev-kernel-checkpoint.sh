#!/bin/bash -l

# a development phase helper script creating a jupyter notebook kernel. 
# kernel matches the environment specified in the training phase job.
# yaml file is extracted by hand from the curated environment "AzureML-sklearn-1.5@latest"
# https://ml.azure.com/registries/azureml/environments/sklearn-1.5/version/5?tid=6f2cb654-d755-4b1d-b2c4-b18e5c3e2442

# TODO: automatically extract the conda requirements file for the curated environment

conda env create -f env/jupyter-sklearn-env.yaml
conda activate sklearn-1.5
python -m ipykernel install --user --name "sklearn-1.5" --display-name "Python (sklearn-1.5)"