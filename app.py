from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.cell_typer_model import CellTyper
from utils.cell_typer_data import CellTyperDataModule
from pytorch_lightning import Trainer, seed_everything
import json


def main(args):

    seed_everything(0)

    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,
        dirpath=args.dirpath,
        filename=args.filename,
        save_top_k=args.save_top_k
    )

    data = CellTyperDataModule(args)
    data.setup('fit')

    vars(args)['training_set_size'] = data.training_set_size
    vars(args)['num_labels'] = data.num_labels
    vars(args)['num_genes'] = data.num_genes
    
    model = CellTyper(args)

    wandb_logger = WandbLogger(
        name=args.wandb_name, 
        project=args.wandb_project
    )

    trainer = Trainer.from_argparse_args(
        args, 
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, data)

    # data.setup('test')

    trainer.test(datamodule=data)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser = CellTyper.add_model_specific_args(parser)
    parser = CellTyperDataModule.add_data_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--wandb_name', type=str, default='CT-test1') # logger
    parser.add_argument('--wandb_project', type=str, default='CellTyperProject') # logger
    parser.add_argument('--monitor', type=str, default='avg_val_loss') # callback
    parser.add_argument('--dirpath', type=str, default=None) # callback
    parser.add_argument('--filename', type=str, default='{epoch}-{avg_val_loss:.2f}') # callback
    parser.add_argument('--save_top_k', type=int, default=1) # callback

    parser.add_argument('--json_args', default=None, help='load arguments from json')
    args = parser.parse_args()
    if args.json_args:
        with open(args.json_args, 'r') as f:
            json_args = Namespace()
            json_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=json_args)
    
    
    main(args)
