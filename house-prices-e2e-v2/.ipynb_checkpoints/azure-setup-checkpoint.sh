#!/bin/sh

# Create random string
guid=$(cat /proc/sys/kernel/random/uuid)
suffix=${guid//[-]/}
suffix=${suffix:0:10}

USER_ID="5160735c-1027-4260-b29b-c0a6569ceab9"
SUB_ID="ad5c3a0e-2e80-43f8-a0e0-2cb6463d9e1e"
EXT_STORAGE_NAME="datastorageacc${suffix}"

az group create --name rg-house-prices-e2e --location westeurope
az configure --defaults group=rg-house-prices-e2e
az ml workspace create --name mlw-house-prices-e2e-zaq12wsx
az configure --defaults workspace=mlw-house-prices-e2e-zaq12wsx
az ml compute create --name ci-zaq12wsx --size STANDARD_DS11_V2 --type ComputeInstance
az ml compute create --name aml-cluster --size STANDARD_DS11_V2 --max-instances 2 --type AmlCompute

# create data blob storage
az storage account create --name $EXT_STORAGE_NAME --sku Standard_LRS --location westeurope
az storage container create --name house-prices-blob-storage --account-name $EXT_STORAGE_NAME
az role assignment create --role "Storage Blob Data Contributor" --assignee $USER_ID --scope "/subscriptions/${SUB_ID}/resourceGroups/rg-house-prices-e2e/providers/Microsoft.Storage/storageAccounts/${EXT_STORAGE_NAME}/blobServices/default/containers/house-prices-blob-storage"