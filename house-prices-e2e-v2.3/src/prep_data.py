import pandas as pd
import argparse
from pathlib import Path

from data_prep_func import clean_df, feat_eng, ft_split, col_flags

def prep_data(train_data_path, test_data_path, data_description_path):

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

    X_joint.loc[train_ix,'split'] = 'train'
    X_joint.loc[test_ix,'split'] = 'test'

    return X_joint, y_joint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type = str)
    parser.add_argument("--test_data", type = str)
    parser.add_argument("--data_description", type = str)
    parser.add_argument("--output_data_path", type = str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    train_data = args.train_data
    test_data = args.test_data
    data_description = args.data_description
    output_data_path = args.ououtput_data_pathtput_path

    X_joint, y_joint = prep_data(train_data, test_data, data_description)

    X_joint.to_csv(Path(output_data_path)/'X_joint.csv', index = False)
    y_joint.to_csv(Path(output_data_path)/'y_joint.csv', index = False)
