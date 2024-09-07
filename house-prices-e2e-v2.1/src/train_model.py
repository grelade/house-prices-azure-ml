#train_model.py

import numpy as np
import pandas as pd
import re
from pathlib import Path
import mlflow
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

from data_prep_func import clean_df, feat_eng, ft_split, col_flags, parse_flags_to_robust, parse_flags_to_minmax
from train_model_func import train_model_nestgrid, train_model

# DATA_DIR = Path('data')
# DATA_DIR = Path('src/data')


parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type = str)
parser.add_argument("--test_data", type = str)
parser.add_argument("--data_description", type = str)
parser.add_argument("--kernel",type=str, default='linear')

args = parser.parse_args()

train_data = args.train_data
test_data = args.test_data
data_description = args.data_description
kernel = args.kernel

# loading train and test data together to ensure all present categories are encoded
df = pd.read_csv(train_data,index_col=0)
df = clean_df(df, data_desc_path = data_description)
df0 = pd.read_csv(test_data,index_col=0)
df0 = clean_df(df0, data_desc_path = data_description)

train_ix = df.index
test_ix = df0.index

df_joint = pd.concat([df,df0],axis=0)

df_joint, cols_feats, cols_target = feat_eng(df_joint,col_flags)
X_joint,y_joint = ft_split(df_joint,cols_feats,cols_target)

X = X_joint.loc[train_ix]
y = y_joint.loc[train_ix]
X_test = X_joint.loc[test_ix]
# y_test = y_joint.loc[test_ix]


cols_robust = parse_flags_to_robust(col_flags)
cols_minmax = parse_flags_to_minmax(col_flags)
cols_impute = list(X_test.columns[X_test.isna().any()])

hparams = {}

with mlflow.start_run():
    mlflow.sklearn.autolog()

    model, mse_train,mse_test = train_model(KernelRidge(kernel = kernel),X,y, 
    cols_robust = cols_robust, 
    cols_minmax = cols_minmax, 
    cols_impute = cols_impute)

    print(model)
    print('MSE train',mse_train)
    print('MSE test',mse_test)

    mlflow.log_metric('mse_test',mse_test)

    y_pred = model.predict(X_test)
    results = X_test.copy()
    results['SalePriceLog'] = y_pred
    results['SalePrice'] = np.exp(results['SalePriceLog'])
    # results = results.reset_index()
    results[['SalePrice']].to_csv('sample_submission.csv')

    mlflow.sklearn.autolog(disable=True)
    mlflow.log_artifact('sample_submission.csv') 