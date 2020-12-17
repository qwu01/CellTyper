from utils.cell_typer_model import CellTyper
from utils.cell_typer_data import CellTyperDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger


# parent_path = "checkpoints"
# parent_path = Path(parent_path)
# models = {}
# for method_path in parent_path.glob("*"):
#     for ckpt_path in method_path.glob("*"):
#         ckpt_path = str(ckpt_path)
#         models[method_path] = CellTyper.load_from_checkpoint(ckpt_path)
# new_trainer = pl.Trainer(resume_from_checkpoint='checkpoints/0_standard/epoch=9-avg_val_loss=0.80.ckpt')
seed_everything(0)

model_test = CellTyper.load_from_checkpoint('checkpoints/0_variational/epoch=29-avg_val_loss=-161639.77.ckpt')
data = CellTyperDataModule(model_test.hparams)
data.setup('fit')
data.setup('test')

checkpoint_callback = ModelCheckpoint(
    monitor=model_test.hparams.monitor,
    dirpath=model_test.hparams.dirpath,
    filename=model_test.hparams.filename,
    save_top_k=model_test.hparams.save_top_k
)

wandb_logger = WandbLogger(
    name=model_test.hparams.wandb_name, 
    project=model_test.hparams.wandb_project
)

trainer = Trainer.from_argparse_args(
    model_test.hparams, 
    logger=wandb_logger,
    callbacks=[checkpoint_callback]
)



trainer.test(model_test, data.test_dataloader())
# print(data)

# test_loader = data.test_dataloader()
# for batch in test_loader:
#     print(batch)
#     print(model_test(batch))

