import mlflow
import argparse
from pathlib import Path


import sys
sys.path.append('..')

from data_prep_func import load_prep_data
from train_model_func import train_model_component, predict_model_component

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep_data_dir", type = str)
    parser.add_argument("--results_dir", type = str)
    parser.add_argument("--kernel",type=str, default='linear')

    args = parser.parse_args()

    return args

def main(args):

    prep_data_dir = args.prep_data_dir
    results_dir = args.results_dir
    model_type = 'kernelridge'
    kernel = args.kernel
    model_params = dict(kernel=kernel)

    with mlflow.start_run():
        mlflow.sklearn.autolog()

        model, mse_train, mse_test = train_model_component(prep_data_dir = prep_data_dir,
                                                           model_type = model_type,
                                                           model_params = model_params)
        _, _, X_test = load_prep_data(prep_data_dir)
        results_test = predict_model_component(model, X_test)

        print(model)
        print('MSE train',mse_train)
        print('MSE test',mse_test)

        mlflow.log_metric('mse_test',mse_test)
        mlflow.sklearn.autolog(disable=True)

        results_dir = Path(results_dir)
        mlflow.sklearn.save_model(model, results_dir / 'model')
        results_test[['SalePrice']].to_csv(results_dir / 'sample_submission.csv')
        # mlflow.log_artifact('sample_submission.csv')

if __name__ == "__main__":

    args = parse_args()

    main(args)
