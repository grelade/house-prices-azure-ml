import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from data_prep_func import col_flags, parse_flags_to_robust, parse_flags_to_minmax
from train_model_func import get_model, train_model


def load_data(input_prep_data_path: str):
    X_joint = pd.read_csv(Path(input_prep_data_path) / 'X_joint.csv')
    y_joint = pd.read_csv(Path(input_prep_data_path) / 'y_joint.csv')

    train_ix = X_joint[X_joint['split'] == 'train'].index
    test_ix = X_joint[X_joint['split'] == 'test'].index

    X = X_joint.loc[train_ix]
    y = y_joint.loc[train_ix]
    X_test = X_joint.loc[test_ix]
    return X, y, X_test

def train_model_pipe(input_prep_data_path,
                     model_type,
                     model_params,
                     ):

    X, y, X_test = load_data(input_prep_data_path)

    cols_robust = parse_flags_to_robust(col_flags)
    cols_minmax = parse_flags_to_minmax(col_flags)
    cols_impute = list(X_test.columns[X_test.isna().any()])

    model = get_model(model_type = model_type)
    model, mse_train,mse_test = train_model(model(**model_params),
                                            X,
                                            y,
                                            cols_robust = cols_robust,
                                            cols_minmax = cols_minmax,
                                            cols_impute = cols_impute)

    y_pred = model.predict(X_test)
    results_test = X_test.copy()
    results_test['SalePriceLog'] = y_pred
    results_test['SalePrice'] = np.exp(results_test['SalePriceLog'])

    return model, mse_train, mse_test, results_test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prep_data_path", type = str)
    parser.add_argument("--kernel",type=str, default='linear')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()

    input_prep_data_path = args.input_prep_data_path

    model_type = 'kernelridge'
    kernel = args.kernel
    model_params = dict(kernel=kernel)

    model, mse_train, mse_test, results_test = train_model_pipe(input_prep_data_path = input_prep_data_path,
                                                                model_type = model_type,
                                                                model_params = model_params)

    print(f'model_type = {model_type}')
    print(f'model_params = {model_params}')
    print('MSE train',mse_train)
    print('MSE test',mse_test)
