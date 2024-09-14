# house-prices-azure-ml
Showcase of various end-to-end ML pipelines using Azure Machine Learning. The versions cover increasingly complex end-to-end solutions to a well-known [House Prices Regression Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) run on Kaggle.

In general, all provided AML solutions have one goal in mind - use Azure ML to take part in the House Prices Regression Competition. To this end, as the input the solution receives training and submission data. The outcome is a submission file with predicted house prices, upload-ready to kaggle competition page.  

Technical details on the versions are given below.

version | dataset | AML framework | MLOps | parameter hypertuning | output | comments |
--------|---------|---------------|-------|-----------------------|--------|----------|
v1.0 | local file | single job | - | sklearn.GridCV | submission file | - |
v1.1 | Data Asset | single job | MLflow | - | submission file | - |
v1.2 | Data Asset | sweep job | MLflow | sweep | submission file | - |
v1.3 | Data Asset | pipeline | MLflow | - | submission file | separate ML and AML logic |
v1.4 | Data Asset | pipeline | MLflow | sweep | submission file + model + upload | output model, auto-upload to kaggle |

