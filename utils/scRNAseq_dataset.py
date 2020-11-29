from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import os
import pandas as pd
from scipy.sparse import save_npz, load_npz

from utils.splitter import generate_split_idx, split_them
from utils.common import MAGIC_NUMBER

class scRNAseq(Dataset):
    def __init__(self, cells, cell_types, genes, expressions):
        self.cells = cells
        self.cell_types = cell_types
        self.genes = genes
        self.expressions = expressions

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, i) -> torch.Tensor:
        #TODO try torch.sparse
        return {
            "expressions": torch.from_numpy(self.expressions[:,i].todense()),
            "cell_types": torch.from_numpy(self.cell_types[i,:].todense()),
            "cell_annotations": self.cells.iloc[i],
            "gene_annotations": self.genes.iloc[i]
        }


class scRNAseqDataModule(pl.LightningDataModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._parse_args()
        self._preprocessing()


    def _parse_args(self):
        self.gene_path = Path(self.args.gene_path)
        self.cell_path = Path(self.args.cell_path)
        self.gene_path = Path(self.args.gene_path)
        self.cell_path = Path(self.args.cell_path)
        self.expression_path = Path(self.args.expression_path)
        self.split_idx_path = Path(self.args.split_idx_path)
        self.split_folder = Path(self.args.split_folder)
        self.batch_size = self.args.batch_size

    def _preprocessing(self):
        # train-test split
        if not os.path.exists(self.split_idx_path):
            generate_split_idx(cell_path=self.cell_path, split_idx_path=self.split_idx_path, reproducible_random_state=MAGIC_NUMBER)
        if not os.path.exists(self.split_folder):
            os.mkdir(self.split_folder) 
            split_them(cell_path=self.cell_path, expression_path=self.expression_path, split_idx_path=self.split_idx_path, split_folder=self.split_folder)

        # readin splited training and test dataset, and gene annotations.
        self.genes = pd.read_csv(self.gene_path, index_col=0)

        self.cells_training = pd.read_csv(self.split_folder/"cells_training.csv", index_col=0)
        self.cells_test = pd.read_csv(self.split_folder/"cells_test.csv", index_col=0)

        self.cell_type_labels_training = load_npz(self.split_folder/"cell_type_labels_training.npz")
        self.cell_type_labels_test = load_npz(self.split_folder/"cell_type_labels_test.npz")

        assert len(self.cells_training) == self.cell_type_labels_training.shape[0], "Something wrong with the training-test split. Confused on row/columns??"
        assert len(self.cells_test) == self.cell_type_labels_test.shape[0], "Something wrong with the training-test split. Confused on row/columns??"

        self.expression_training = load_npz(self.split_folder/"expression_training.npz")
        self.expression_test = load_npz(self.split_folder/"expression_test.npz")

        assert self.expression_training.shape[1] == self.cell_type_labels_training.shape[0], "Something wrong with the training-test split. Confused on row/columns??"
        assert self.expression_test.shape[1] == self.cell_type_labels_test.shape[0], "Something wrong with the training-test split. Confused on row/columns??"


    def prepare_data(self):
        self.examples_training = scRNAseq(cells=self.cells_training, cell_types=self.cell_type_labels_training, genes=self.genes, expressions=self.expression_training)
        self.examples_test = scRNAseq(cells=self.cells_test, cell_types=self.cell_type_labels_test, genes=self.genes, expressions=self.expression_test)
        
    def train_dataloader(self):
        return DataLoader(dataset=self.examples_training, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.examples_test, batch_size=self.batch_size, shuffle=True)