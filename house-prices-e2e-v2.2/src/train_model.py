import numpy as np
import pandas as pd

from data_prep_func import clean_df, feat_eng, ft_split, col_flags, parse_flags_to_robust, parse_flags_to_minmax
from train_model_func import get_model, train_model

def train_model_pipe(model_type,
                     model_params,
                     train_data_path,
                     test_data_path,
                     data_description_path):

    # loading train and test data together to ensure all present categories are encoded
    df = pd.read_csv(train_data_path,index_col=0)
    df = clean_df(df, data_desc_path = data_description_path)
    df0 = pd.read_csv(test_data_path,index_col=0)
    df0 = clean_df(df0, data_desc_path = data_description_path)

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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type = str)
    parser.add_argument("--test_data", type = str)
    parser.add_argument("--data_description", type = str)
    parser.add_argument("--kernel",type=str, default='linear')

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    data_description = args.data_description

    model_type = 'kernelridge'
    kernel = args.kernel
    model_params = dict(kernel=kernel)

    model, mse_train, mse_test, results_test = train_model_pipe(model_type = model_type,
                                                 model_params = model_params,
                                                 train_data_path = train_data,
                                                 test_data_path = test_data,
                                                 data_description_path = data_description)

    print(f'model_type = {model_type}')
    print(f'model_params = {model_params}')
    print('MSE train',mse_train)
    print('MSE test',mse_test)
