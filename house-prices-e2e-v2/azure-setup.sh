#!/bin/sh

# Create random string
guid=$(cat /proc/sys/kernel/random/uuid)
suffix=${guid//[-]/}
suffix=${suffix:0:10}

az group create --name rg-house-prices-e2e --location westeurope
az configure --defaults group=rg-house-prices-e2e
az ml workspace create --name mlw-house-prices-e2e-zaq12wsx
az configure --defaults workspace=mlw-house-prices-e2e-zaq12wsx
az ml compute create --name ci-zaq12wsx --size STANDARD_DS11_V2 --type ComputeInstance
az ml compute create --name aml-cluster --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute




