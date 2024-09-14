#data_prep_func.py

import numpy as np
import pandas as pd
from enum import Flag, auto
import re
from pathlib import Path

# import mltable

def get_data_desc_dict(data_desc_path = 'data_description.txt'):

    with open(data_desc_path,'r') as fp:
        lines = fp.readlines()

    desc_dict = {}
    col_name = None
    for l in lines:
        if len(l.split(': ')) == 2 and l.count('\t')==0:
            ls = l.split(': ')
            col_name = ls[0]
            desc_dict[col_name] = {'desc':ls[1].strip(),'rename_dict':{}}

        if col_name:
            if l[:7] == '       ' and len(l.strip())>0:

                ls = l.split('\t')
                df_value = ls[0]
                new_df_value = ls[1].strip()

                if col_name in ['MSSubClass','OverallQual','OverallCond']:
                    df_value = int(df_value)
                # if col_name == 'MSSubClass':
                #     df_value = int(df_value)
                else:
                    df_value = df_value.strip()

                # if col_name in []:
                #     continue

                desc_dict[col_name]['rename_dict'][df_value] = new_df_value

    desc_dict['SalePrice'] = {'desc':'Sale Price','rename_dict':{}}

    return desc_dict

# compare desc data with df ; identify inconsistencies
def print_desc_data_comparison(df,desc_dict):

    print('column name \t n_desc_values \t n_df_values \t missing_df_values')
    for col in df.columns:
        n_desc_values = len(desc_dict[col]['rename_dict'])
        n_df_values = len(df[col].unique())
        missing_df_values = [val for val in df[col].unique() if val not in desc_dict[col]['rename_dict'].keys()]

        is_correct = n_desc_values==0 or n_desc_values >= n_df_values
        if n_desc_values > 0:
            print(f'{col:10} \t {n_desc_values:1} \t {n_df_values:10} \t \t {missing_df_values}')

def rename_wrong_values(df):
    df.MSZoning = np.where(df.MSZoning == 'C (all)','C',df.MSZoning)

    df.Neighborhood = np.where(df.Neighborhood == 'NAmes','Names',df.Neighborhood)

    df.BldgType = np.where(df.BldgType == '2fmCon','2FmCon',df.BldgType)
    df.BldgType = np.where(df.BldgType == 'Duplex','Duplx',df.BldgType)
    df.BldgType = np.where(df.BldgType == 'Twnhs','TwnhsI',df.BldgType)

    df.Exterior2nd = np.where(df.Exterior2nd == 'Wd Shng','Wd Sdng',df.Exterior2nd)
    df.Exterior2nd = np.where(df.Exterior2nd == 'CmentBd','CemntBd',df.Exterior2nd)
    df.Exterior2nd = np.where(df.Exterior2nd == 'Brk Cmn','BrkComm',df.Exterior2nd)

    return df

# filling in NANs
def fill_nans(df):
    # not sure about this one
    df.LotFrontage = df.LotFrontage.fillna(0)

    df.Alley = df.Alley.fillna('NA')
    df.MasVnrType = df.MasVnrType.fillna('None')
    df.MasVnrArea = df.MasVnrArea.fillna(0)

    df[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']] = df[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']].fillna('NA')
    df.FireplaceQu = df.FireplaceQu.fillna('NA')

    df.GarageType = df.GarageType.fillna('NA')
    df.GarageYrBlt = df.GarageYrBlt.fillna(0)
    df.GarageFinish = df.GarageFinish.fillna('NA')
    df.GarageQual = df.GarageQual.fillna('NA')
    df.GarageCond = df.GarageCond.fillna('NA')

    df.PoolQC = df.PoolQC.fillna('NA')
    df.Fence = df.Fence.fillna('NA')
    df.MiscFeature = df.MiscFeature.fillna('NA')

    df = df.dropna(subset=['Electrical'])
    return df

def incorp_desc_dict_to_df(df,desc_dict):
    df = df.copy()
    for col in desc_dict.keys():

        rename_dict = desc_dict[col]['rename_dict']
        if len(rename_dict)>0:
            df[col] = df[col].map(rename_dict)
        # break

    return df

# cast into correct types

def cast_categorical_vars(df,desc_dict):

    df = df.astype({'MSSubClass':'category',
                    'MSZoning':'category',
                    # 'LotFrontage':'int',
                    'Street':'category',
                    'Alley':'category',
                    'LotShape':'category', #ordinal
                    'LandContour':'category',
                    'Utilities':'category', #ordinal
                    'LotConfig':'category',
                    'LandSlope':'category', #ordinal
                    'Neighborhood':'category',
                    'Condition1':'category',
                    'Condition2':'category',
                    'BldgType':'category',
                    'HouseStyle':'category',
                    'OverallQual':'category', #ordinal
                    'OverallCond':'category', #ordinal
                    'RoofStyle':'category',
                    'RoofMatl':'category',
                    'Exterior1st':'category',
                    'Exterior2nd':'category',
                    'MasVnrType':'category',
                    'MasVnrArea':'int',
                    'ExterQual':'category', #ordinal
                    'ExterCond':'category', #ordinal
                    'Foundation':'category',
                    'BsmtQual': 'category', #ordinal
                    'BsmtCond':'category', #ordinal
                    'BsmtExposure':'category', #ordinal
                    'BsmtFinType1':'category', #ordinal
                    'BsmtFinType2':'category', #ordinal
                    'Heating':'category',
                    'HeatingQC':'category', #ordinal
                    'CentralAir':'category',
                    'Electrical':'category', #ordinal
                    'KitchenQual':'category', #ordinal
                    'Functional':'category',
                    'FireplaceQu':'category', #ordinal
                    'GarageType':'category',
                    'GarageYrBlt':'int',
                    'GarageFinish':'category', #ordinal
                    'GarageQual':'category', #ordinal
                    'GarageCond':'category', #ordinal
                    'PavedDrive':'category',
                    'PoolQC':'category', #ordinal
                    'Fence':'category', #ordinal
                    'MiscFeature':'category',
                    'SaleType':'category',
                    'SaleCondition':'category'
                   })

    # setup ordered categories

    ordinal_columns = ['LotShape','Utilities','LandSlope',
                       'OverallQual','OverallCond','ExterQual',
                       'BsmtQual','BsmtCond','BsmtExposure',
                       'BsmtFinType1','BsmtFinType2','HeatingQC',
                       'Electrical','KitchenQual','FireplaceQu',
                       'GarageFinish','GarageQual','GarageCond',
                       'PoolQC','Fence']

    for col in ordinal_columns:
        categories = list(desc_dict[col]['rename_dict'].values())[::-1]
        # print(len(categories),len(df[col].cat.categories))
        df[col] = df[col].cat.as_ordered()
        df[col] = df[col].cat.set_categories(categories)

    # complete unordered columns with desc_dict data
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            cat_names = list(desc_dict[col]['rename_dict'].values())
            if col not in ordinal_columns and len(df[col].cat.categories) != len(cat_names):
                df[col] = df[col].cat.set_categories(cat_names)
    return df

def clean_df(df,data_desc_path = 'data_description.txt', verbose=False):

    desc_dict = get_data_desc_dict(data_desc_path = data_desc_path)
    if verbose: print_desc_data_comparison(df,desc_dict)

    df = rename_wrong_values(df)
    df = fill_nans(df)
    if verbose: print_desc_data_comparison(df,desc_dict)

    df = incorp_desc_dict_to_df(df,desc_dict)

    df = cast_categorical_vars(df,desc_dict)
    return df



from enum import Flag, auto

# FLAGS
class coltype(Flag):
    SKIP = auto() # skipped
    CAT = auto() # considered categorical
    NUM = auto() # considered numeric
    CATNUM = auto() # categorical cast to numeric
    LOG = auto() # transform to log feature
    NONZERO = auto() # add nonzero feature
    MINMAX = auto() # scaler type
    ROBUST = auto() # scaler type

col_flags =  {'MSSubClass': coltype.SKIP,
              'MSZoning': coltype.CAT,
              'LotFrontage': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'LotArea': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'Street': coltype.SKIP,
              'Alley': coltype.CAT,
              'LotShape': coltype.CAT,
              'LandContour': coltype.CAT,
              'Utilities': coltype.SKIP,
              'LotConfig': coltype.CAT,
              'LandSlope': coltype.CAT,
              'Neighborhood': coltype.CAT,
              'Condition1': coltype.SKIP, # double columns
              'Condition2': coltype.SKIP, # double columns
              'BldgType': coltype.CAT,
              'HouseStyle': coltype.CAT,
              'OverallQual': coltype.CATNUM | coltype.MINMAX,
              'OverallCond': coltype.CATNUM | coltype.MINMAX,
              'YearBuilt': coltype.NUM | coltype.LOG | coltype.ROBUST,
              'YearRemodAdd': coltype.NUM | coltype.LOG | coltype.ROBUST,
              'RoofStyle': coltype.CAT,
              'RoofMatl': coltype.SKIP,
              'Exterior1st': coltype.SKIP, # double columns
              'Exterior2nd': coltype.SKIP, # double columns
              'MasVnrType': coltype.CAT,
              'MasVnrArea': coltype.NUM | coltype.LOG | coltype.ROBUST,
              'ExterQual': coltype.CATNUM | coltype.MINMAX,
              'ExterCond': coltype.CATNUM | coltype.MINMAX,
              'Foundation': coltype.CAT,
              'BsmtQual': coltype.CATNUM | coltype.MINMAX,
              'BsmtCond': coltype.CATNUM | coltype.MINMAX,
              'BsmtExposure': coltype.CAT,
              'BsmtFinType1': coltype.SKIP, # double columns
              'BsmtFinSF1': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'BsmtFinType2': coltype.SKIP, # double columns
              'BsmtFinSF2': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'BsmtUnfSF': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'TotalBsmtSF': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'Heating': coltype.SKIP,
              'HeatingQC': coltype.CATNUM | coltype.MINMAX,
              'CentralAir': coltype.CAT,
              'Electrical': coltype.CAT,
              '1stFlrSF': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              '2ndFlrSF': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'LowQualFinSF': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'GrLivArea': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'BsmtFullBath': coltype.NUM | coltype.MINMAX,
              'BsmtHalfBath': coltype.NUM | coltype.MINMAX,
              'FullBath': coltype.NUM | coltype.MINMAX,
              'HalfBath': coltype.NUM | coltype.MINMAX,
              'BedroomAbvGr': coltype.NUM | coltype.MINMAX,
              'KitchenAbvGr': coltype.NUM | coltype.MINMAX,
              'KitchenQual': coltype.CATNUM | coltype.MINMAX,
              'TotRmsAbvGrd': coltype.NUM | coltype.MINMAX,
              'Functional': coltype.SKIP,
              'Fireplaces': coltype.NUM | coltype.MINMAX,
              'FireplaceQu': coltype.CATNUM | coltype.MINMAX,
              'GarageType': coltype.CAT,
              'GarageYrBlt': coltype.NUM | coltype.LOG | coltype.ROBUST,
              'GarageFinish': coltype.CATNUM | coltype.MINMAX,
              'GarageCars': coltype.NUM | coltype.MINMAX,
              'GarageArea': coltype.NUM | coltype.LOG | coltype.ROBUST,
              'GarageQual': coltype.CATNUM | coltype.MINMAX,
              'GarageCond': coltype.CATNUM | coltype.MINMAX,
              'PavedDrive': coltype.CAT,
              'WoodDeckSF': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'OpenPorchSF': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'EnclosedPorch': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              '3SsnPorch': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'ScreenPorch': coltype.NUM | coltype.LOG | coltype.NONZERO | coltype.ROBUST,
              'PoolArea': coltype.NONZERO,
              'PoolQC': coltype.SKIP,
              'Fence': coltype.CAT,
              'MiscFeature': coltype.SKIP,
              'MiscVal' : coltype.SKIP,
              'MoSold': coltype.NUM | coltype.MINMAX,
              'YrSold': coltype.NUM | coltype.LOG | coltype.ROBUST,
              'SaleType': coltype.SKIP,
              'SaleCondition': coltype.SKIP,
              'SalePrice': coltype.NUM | coltype.LOG,
             }


def parse_flags_to_form_nonzero(flags):
    return [col for col,flag in flags.items() if coltype.NONZERO in flag]

def parse_flags_to_form_log(flags):
    return [col for col,flag in flags.items() if coltype.LOG in flag]

def parse_flags_to_cat_as_num(flags):
    return [col for col,flag in flags.items() if coltype.CATNUM in flag]

def parse_flags_to_num(flags, exclude_log = False):
    cols_num = [col for col,flag in flags.items() if coltype.NUM in flag]
    if exclude_log:
        cols_num = [col for col in cols_num if col not in parse_flags_to_form_log(flags)]
    return cols_num

def parse_flags_to_cat(flags, exclude_cat_as_num = False):
    cols_cat = [col for col,flag in flags.items() if coltype.CAT in flag]
    if exclude_cat_as_num:
        cols_cat = [col for col in cols_cat if col not in parse_flags_to_cat_as_num(flags)]

    return cols_cat

def parse_flags_to_skip(flags):
    return [col for col,flag in flags.items() if coltype.SKIP in flag]

def parse_flags_to_minmax(flags):
    cols_minmax = [col for col,flag in flags.items() if coltype.MINMAX in flag]
    return cols_minmax

def parse_flags_to_robust(flags):
    cols_robust = [col for col,flag in flags.items() if coltype.ROBUST in flag]
    cols_log = parse_flags_to_form_log(flags)
    cols_robust = [col+'Log' if col in cols_log else col for col in cols_robust]
    return cols_robust

def feat_eng_nonzero(df, cols):
    '''
    create binary columns
    '''

    df[[col+'Nonzero' for col in cols]] = (df[cols] > 0)*1.
    return df

def feat_eng_log(df, cols, eps = 1e-10):
    '''
    create log columns
    '''

    df[[col+'Log' for col in cols]] = np.log(df[cols]+eps)
    return df

def feat_eng_cat_as_num(df, cols):
    '''
    transform cat columns to numeric
    '''

    df[cols] = df[cols].apply(lambda colvars: colvars.cat.codes.astype('float'))
    return df

def feat_eng_cat(df, cols, drop_first = True):
    '''
    onehot encode categorical columns
    '''

    df = pd.get_dummies(df, columns=cols,drop_first = drop_first,dtype=float)
    return df

def feat_eng(df,col_flags, output_col = 'SalePriceLog'):
    df = df.copy()
    cols_input = []
    cols_output = []

    # form nonzero binary columns
    cols = parse_flags_to_form_nonzero(col_flags)
    df = feat_eng_nonzero(df,cols)
    cols_input += [col+'Nonzero' for col in cols]

    # form log transforms
    cols = parse_flags_to_form_log(col_flags)
    df = feat_eng_log(df,cols)
    cols_input += [col+'Log' for col in cols]

    # transform cat to num
    cols = parse_flags_to_cat_as_num(col_flags)
    df = feat_eng_cat_as_num(df,cols)
    cols_input += cols

    # numeric columns
    cols = parse_flags_to_num(col_flags, exclude_log = True)
    df[cols] = df[cols].astype(float)
    cols_input += cols

    # # onehot encode cat columns
    cols = parse_flags_to_cat(col_flags, exclude_cat_as_num = True)
    df = feat_eng_cat(df,cols)
    cols_input += [col for col in df.columns if col.split('_')[0] in cols]


    cols_output = list(filter(lambda x: x == output_col,cols_input))
    cols_input = list(filter(lambda x: x != output_col,cols_input))

    return df, cols_input, cols_output

def ft_split(df,cols_feats,cols_target):
    X = df[cols_feats]
    # y = None
    # if len(cols_target) == 1:
    y = df[cols_target]

    new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in X.columns}
    X = X.rename(new_names,axis=1)

    return X, y


def prep_data_component(train_data_path: str, test_data_path: str, data_description_path: str):

    # loading train and test data together to ensure all present categories are encoded

    # df = mltable.from_delimited_files(paths=[{'file': train_data_path}]).to_pandas_dataframe()
    df = pd.read_csv(train_data_path)
    df = clean_df(df, data_desc_path = data_description_path)
    df['split'] = 'train'

    # df0 = mltable.from_delimited_files(paths=[{'file': test_data_path}]).to_pandas_dataframe()
    df0 = pd.read_csv(test_data_path)
    df0 = clean_df(df0, data_desc_path = data_description_path)
    df0['split'] = 'test'

    df_joint = pd.concat([df,df0],axis=0)
    df_joint = df_joint.reset_index(drop=True)

    df_joint, cols_feats, cols_target = feat_eng(df_joint,col_flags)

    X_joint,y_joint = ft_split(df_joint,['Id','split']+cols_feats,['Id','split']+cols_target)
    return X_joint, y_joint

def load_prep_data(input_prep_data_path: str):
    X_joint = pd.read_csv(Path(input_prep_data_path) / 'X_joint.csv',index_col=0)
    y_joint = pd.read_csv(Path(input_prep_data_path) / 'y_joint.csv',index_col=0)

    train_ix = X_joint[X_joint['split'] == 'train'].index
    test_ix = X_joint[X_joint['split'] == 'test'].index

    X = X_joint.loc[train_ix]
    y = y_joint.loc[train_ix]
    X_test = X_joint.loc[test_ix]

    X = X.drop('split',axis=1)
    X = X.set_index('Id')
    y = y[['Id','SalePriceLog']].set_index('Id')
    X_test = X_test.drop('split',axis=1)
    X_test = X_test.set_index('Id')

    return X, y, X_test

def save_prep_data(X_joint,y_joint, output_data_dir = '.'):
    X_joint.to_csv(Path(output_data_dir)/'X_joint.csv')
    y_joint.to_csv(Path(output_data_dir)/'y_joint.csv')
