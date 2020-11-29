from pathlib import Path
from scipy.io import mmread as readMM
import json, os
from argparse import ArgumentParser, Namespace
import pandas as pd




def main(args):
    pass
    # 1. check if the dataset has been split or not.
        # If not, split into training and validation datasets.

        # initiate pl.DataModule
        # initiate model

    # 2. train.

    # 3. evaluate.



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--json_args', help='load arguments from json')
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.json_args:
        with open(args.json_args, 'r') as f:
            json_args = Namespace()
            json_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=json_args)

    main(args)


