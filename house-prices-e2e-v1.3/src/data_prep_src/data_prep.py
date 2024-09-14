import argparse

import sys
sys.path.append('..')

from data_prep_func import prep_data_component, save_prep_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type = str)
    parser.add_argument("--test_data", type = str)
    parser.add_argument("--data_description", type = str)
    parser.add_argument("--output_data_dir", type = str)

    args = parser.parse_args()
    return args

def main(args):

    train_data = args.train_data
    test_data = args.test_data
    data_description = args.data_description
    output_data_dir = args.output_data_dir

    X_joint, y_joint = prep_data_component(train_data, test_data, data_description)
    save_prep_data(X_joint,y_joint, output_data_dir = output_data_dir)

if __name__ == "__main__":

    args = parse_args()

    main(args)
