import mlflow
import argparse

from train_model import train_model_pipe

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


with mlflow.start_run():
    mlflow.sklearn.autolog()

    model, mse_train, mse_test, results_test = train_model_pipe(model_type = model_type,
                                                 model_params = model_params,
                                                 train_data_path = train_data,
                                                 test_data_path = test_data,
                                                 data_description_path = data_description)

    print(model)
    print('MSE train',mse_train)
    print('MSE test',mse_test)

    mlflow.log_metric('mse_test',mse_test)

    mlflow.sklearn.autolog(disable=True)
    results_test[['SalePrice']].to_csv('sample_submission.csv')
    mlflow.log_artifact('sample_submission.csv')
