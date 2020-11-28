from pathlib import Path
from scipy.io import mmread as readMM
import json
from argparse import ArgumentParser, Namespace
import pandas as pd


def main(args):
    gene_path = Path(args.gene_path)
    cell_path = Path(args.cell_path)
    expression_path = Path(args.expression_path)

    genes = pd.read_csv(gene_path)
    print(genes)
    cells = pd.read_csv(cell_path)
    print(cells)
    tmp_exp = readMM(expression_path)
    tmp_exp = tmp_exp.tocsr()
    print(type(tmp_exp))



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


