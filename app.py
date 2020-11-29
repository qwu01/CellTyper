from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import json


from utils.cell_typer_dataset import CellTyperDataModule
from utils.cell_typer_model import CellTyper


def main(args):
    
    # 1. check if the dataset has been split or not.
        # If not, split into training and validation datasets.

        # initiate pl.DataModule
        # initiate model

    # 2. train.

    # 3. evaluate.

    data_module = CellTyperDataModule(args)
    model = CellTyper(args)
    wandb_logger = WandbLogger(name=args.wandb_name, project=args.wandb_project)
    trainer = pl.Trainer(logger=wandb_logger, gpus=-1, accumulate_grad_batches=args.accumulate_grad_batches, max_epochs=2, log_every_n_steps=args.log_every_n_steps, default_root_dir = args.default_root_dir)
    trainer.fit(model, data_module)



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


