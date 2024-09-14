# house-prices-azure-ml
Showcase of various end-to-end ML pipelines using Azure Machine Learning. The versions cover increasingly complex end-to-end solutions to a well-known [House Prices Regression Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) run on Kaggle.

In general, all provided AML solutions have one goal in mind - use Azure ML to take part in the House Prices Regression Competition. To this end, as the input the solution receives training and submission data. The outcome is a submission file with predicted house prices, upload-ready to kaggle competition page.  

Technical details on the versions are given below.

version | dataset | AML framework | MLOps | parameter hypertuning | output | comments |
--------|---------|---------------|-------|-----------------------|--------|----------|
house-prices-e2e-v1 | local file | single job | - | sklearn.GridCV | submission file | - |
house-prices-e2e-v2 | Data Asset | single job | MLflow | - | submission file | - |
house-prices-e2e-v2.1 | Data Asset | sweep job | MLflow | sweep | submission file | - |
house-prices-e2e-v2.3 | Data Asset | pipeline | MLflow | - | submission file | separate ML and AML logic |
house-prices-e2e-v2.4 | Data Asset | pipeline | MLflow | - | submission file + model | pipeline-compatible refactor |
house-prices-e2e-v2.5 | Data Asset | pipeline | MLflow | - | submission file + model + upload | auto-upload to kaggle |
house-prices-e2e-v2.6 | Data Asset | pipeline | MLflow | sweep | submission file + model + upload | - |

