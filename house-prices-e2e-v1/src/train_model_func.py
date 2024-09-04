#train_model_func.py

import numpy as np

from sklearn.metrics import mean_squared_error as MSE

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import VotingRegressor
from sklearn.impute import KNNImputer

from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.ensemble import StackingRegressor

from sklearn import set_config
set_config(transform_output = "pandas")

# import lightgbm as lgbm

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)


def get_custom_scaler(cols_robust, 
                      cols_minmax, **kwargs):
    transformers = [('robust_scaler', RobustScaler(), cols_robust),
                    ('minmax_scaler', MinMaxScaler(), cols_minmax)]
    scaler = ColumnTransformer(transformers,
                               remainder='passthrough',verbose_feature_names_out=False)

    return scaler

def get_custom_imputer(cols_impute,**kwargs):
    transformers = [('imputer',KNNImputer(),cols_impute)]
    imputer = ColumnTransformer(transformers,
                                remainder='passthrough',verbose_feature_names_out=False)
    return imputer

def train_model_cv(model,X,y, random_state = 4255,**kwargs):
    pipeline = Pipeline([('imputer',get_custom_imputer(**kwargs)),
                         ('scaler', get_custom_scaler(**kwargs)),
                         ('model', model)])
    
    kf = KFold(5,shuffle=True,random_state = random_state)
    
    score = cross_val_score(pipeline,X,y,scoring='neg_mean_squared_error',cv=kf)
    mse = -np.mean(score)
    pipeline.fit(X,y)
    
    return pipeline, mse
    
def train_model_grid(model,X,y,hparams, random_state = 4255, **kwargs):
    
    # exp_transformer = FunctionTransformer(np.exp, validate=True)
    pipeline = Pipeline([('imputer',get_custom_imputer(**kwargs)),
                         ('scaler', get_custom_scaler(**kwargs)),
                         ('model', model)])
    
    kf = KFold(5,shuffle=True,random_state = random_state)
    
    grid = GridSearchCV(pipeline,param_grid=hparams,scoring='neg_mean_squared_error',cv=kf)
    result = grid.fit(X,y)
    
    mse = -result.best_score_
    best_model = result.best_estimator_
    return best_model,mse

def train_model_nestgrid(model,X,y,hparams, random_state = 4255, **kwargs):
    pipeline = Pipeline([('imputer',get_custom_imputer(**kwargs)),
                             ('scaler', get_custom_scaler(**kwargs)),
                             ('model', model)])

    inner_fold = KFold(5,shuffle=True,random_state=random_state)
    grid = GridSearchCV(pipeline,
                        param_grid = hparams,
                        scoring = 'neg_mean_squared_error',
                        cv = inner_fold,
                        refit=True)
    
    outer_fold = KFold(5,shuffle=True,random_state=random_state) 
    cv_results = cross_validate(grid,
                                X,y,
                                cv = outer_fold,
                                scoring = 'neg_mean_squared_error',
                                return_train_score = True, 
                                return_estimator = True)
    
    best_model = grid.fit(X,y)
    mse_train = - cv_results['train_score'].mean()
    mse_test = - cv_results['test_score'].mean()
    
    # return cv_results
    return best_model, mse_train, mse_test