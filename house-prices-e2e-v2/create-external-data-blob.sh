#!/bin/sh

# Create random string
guid=$(cat /proc/sys/kernel/random/uuid)
suffix=${guid//[-]/}
suffix=${suffix:0:10}

# create data blob storage
TENANT_ID="6f2cb654-d755-4b1d-b2c4-b18e5c3e2442"
USER_ID="5160735c-1027-4260-b29b-c0a6569ceab9"
SUB_ID="ad5c3a0e-2e80-43f8-a0e0-2cb6463d9e1e"
EXT_STORAGE_NAME="datastorageacc${suffix}"

az storage account create --name $EXT_STORAGE_NAME --sku Standard_LRS --location westeurope
az storage container create --name house-prices-blob-storage --account-name $EXT_STORAGE_NAME
az role assignment create --role "Storage Blob Data Contributor" --assignee $USER_ID --scope "/subscriptions/${SUB_ID}/resourceGroups/rg-house-prices-e2e/providers/Microsoft.Storage/storageAccounts/${EXT_STORAGE_NAME}/blobServices/default/containers/house-prices-blob-storage"
azcopy --login --tenant-id $TENANT_ID
azcopy copy data/ "https://${EXT_STORAGE_NAME}.blob.core.windows.net/house-prices-blob-storage" --recursive