EXT_STORAGE_NAME=""
azcopy --login --tenant-id 6f2cb654-d755-4b1d-b2c4-b18e5c3e2442
azcopy copy data/ https://${EXT_STORAGE_NAME}.blob.core.windows.net/house-prices-blob-storage --recursive
