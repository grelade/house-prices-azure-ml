import mlflow
import argparse
from pathlib import Path

import sys
sys.path.append('..')

from data_prep_func import prep_data_component, save_prep_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str)
    parser.add_argument("--prep_data_dir", type = str)

    args = parser.parse_args()
    return args

def main(args):

    data_dir = args.data_dir
    prep_data_dir = args.prep_data_dir

    # mlflow.log_param('data_dir',data_dir)
    # print('[LOG] data_dir', data_dir)
    # print('[LOG] prep_data_dir', prep_data_dir)

    train_data = Path(data_dir) / 'train.csv'
    test_data = Path(data_dir) / 'test.csv'
    data_description = Path(data_dir) / 'data_description.txt'

    X_joint, y_joint = prep_data_component(train_data, test_data, data_description)
    save_prep_data(X_joint,y_joint, output_data_dir = prep_data_dir)

if __name__ == "__main__":

    args = parse_args()

    main(args)
