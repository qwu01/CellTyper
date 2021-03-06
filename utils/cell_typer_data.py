import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from scipy.io import mmread as readMM
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CellTyperDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(CellTyperDataModule, self).__init__()
        self.args = args
        self.get_dummies = OneHotEncoder(handle_unknown="ignore")
        self.standard_scaler = StandardScaler()
        self.folders = {
            "training": Path(self.args.training_set_folder),
            "validation": Path(self.args.validation_set_folder),
            "test": Path(self.args.test_set_folder)
        }
        self.genes = pd.read_csv(self.folders['training']/"gene.csv", index_col=0)

        if self.args.feature_selection_method:
            if self.args.feature_selection_method == "random":
                col_name = "rand" + str(self.args.feature_number)
            elif self.args.feature_selection_method == "scmap":
                col_name = "scmap" + str(self.args.feature_number)
            elif self.args.feature_selection_method == "hvg":
                col_name = "hvg" + str(self.args.feature_number)
            else:
                raise Exception("Check feature selection method in .json")

            self.gene_selection_mask = pd.read_csv(self.folders['training']/"gene_selection.csv", index_col=0)
            assert all(self.genes.index == self.gene_selection_mask.index), "Feature selection mask index is not aligned to expression matrix!!"
            self.gene_selection_mask = self.gene_selection_mask[col_name].values

            self.genes = self.genes[self.gene_selection_mask] # in case need the gene information, call dataset.genes
            

        self.cells = {
            "training": pd.read_csv(self.folders["training"]/"cell.csv", index_col=0).reset_index(drop=True)[['Name', 'CellType']],
            "validation": pd.read_csv(self.folders["validation"]/"cell.csv", index_col=0).reset_index(drop=True)[['Name', 'CellType']],
            "test": pd.read_csv(self.folders["test"]/"cell.csv", index_col=0).reset_index(drop=True)[['Name', 'CellType']]
        }
        
        self.get_dummies.fit(self.cells["training"][["CellType"]])

    def setup(self, stage=None):

        if stage == "fit" or stage is None:

            cell_type_training = torch.Tensor(self.get_dummies.transform(self.cells["training"][['CellType']]).toarray()).type(torch.FloatTensor)
            cell_type_validation = torch.Tensor(self.get_dummies.transform(self.cells["validation"][['CellType']]).toarray()).type(torch.FloatTensor)

            positive_weights_training = (cell_type_training.shape[0] - cell_type_training.sum(axis=0))/cell_type_training.sum(axis=0)
            positive_weights_validation = (cell_type_validation.shape[0] - cell_type_validation.sum(axis=0))/cell_type_validation.sum(axis=0)

            expression_training = readMM(self.folders["training"]/"log_norm_count_sparse.mtx").tocsr()
            expression_training = expression_training.todense()
            if self.args.feature_selection_method:
                expression_training = expression_training[self.gene_selection_mask]
            expression_training = torch.tensor(expression_training).type(torch.FloatTensor)
            expression_training = torch.transpose(expression_training, 0, 1)

            expression_validation = readMM(self.folders["validation"]/"log_norm_count_sparse.mtx").tocsr()
            expression_validation = expression_validation.todense()
            if self.args.feature_selection_method:
                expression_validation = expression_validation[self.gene_selection_mask]
            expression_validation = torch.tensor(expression_validation).type(torch.FloatTensor)
            expression_validation = torch.transpose(expression_validation, 0, 1)

            self.standard_scaler.fit(expression_training)
            expression_training = torch.Tensor(self.standard_scaler.transform(expression_training))
            expression_validation = torch.Tensor(self.standard_scaler.transform(expression_validation))

            self.training_set = CellTyperDataSet(cell_type_training, expression_training, positive_weights_training)
            self.validation_set = CellTyperDataSet(cell_type_validation, expression_validation, positive_weights_validation)
            self.training_set_size = self.training_set.n_examples

            self.num_labels = self.training_set.n_labels
            self.num_genes = self.training_set.n_features

        if stage == "test" or stage is None:

            cell_type_test = torch.Tensor(self.get_dummies.transform(self.cells["test"][['CellType']]).toarray()).type(torch.FloatTensor)

            positive_weights_test = (cell_type_test.shape[0] - cell_type_test.sum(axis=0))/cell_type_test.sum(axis=0)

            expression_test = readMM(self.folders["test"]/"log_norm_count_sparse.mtx").tocsr()
            expression_test = expression_test.todense()
            if self.args.feature_selection_method:
                expression_test = expression_test[self.gene_selection_mask]
            expression_test = torch.tensor(expression_test).type(torch.FloatTensor)
            expression_test = torch.transpose(expression_test, 0, 1)

            expression_test = torch.Tensor(self.standard_scaler.transform(expression_test))

            self.test_set = CellTyperDataSet(cell_type_test, expression_test, positive_weights_test)

    def train_dataloader(self):
        training_data_loader = DataLoader(dataset=self.training_set, batch_size=self.args.batch_size)
        return training_data_loader
    
    def val_dataloader(self):
        val_data_loader = DataLoader(dataset=self.validation_set, batch_size=self.args.batch_size)
        return val_data_loader
    
    def test_dataloader(self):
        test_data_loader = DataLoader(dataset=self.test_set, batch_size=self.args.batch_size)
        return test_data_loader

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--training_set_folder", type=str, default="Data/datasets/pbmc1_10x_Chromium_v2_A_training/") # for later using different set of training data
        parser.add_argument("--validation_set_folder", type=str, default="Data/datasets/pbmc1_10x_Chromium_v2_A_validation") # for later using different set of training data
        parser.add_argument("--test_set_folder", type=str, default="Data/datasets/pbmc1_10x_Chromium_v2_A_test") # for later using different set of training data
        parser.add_argument("--batch_size", type=int, default=4000)

        parser.add_argument("--feature_selection_method", type=str, default=None,  choices=("scmap", "hvg", "random"))
        parser.add_argument("--feature_number", type=int, default=None, choices=(300,500,1000,2000))
        return parser


class CellTyperDataSet(Dataset):
    def __init__(self, cell_types, expressions, positive_weights):
        assert cell_types.shape[0] == expressions.shape[0]
        assert cell_types.shape[1] == positive_weights.shape[0]
        self.cell_types = cell_types
        self.expressions = expressions
        self.positive_weights = positive_weights
        self.n_examples = self.cell_types.shape[0]
        self.n_labels = self.cell_types.shape[1]
        self.n_features = self.expressions.shape[1]

    def __len__(self):
        return self.n_examples

    def __getitem__(self, i):
        expressions = self.expressions[i,:].type(torch.FloatTensor)
        cell_types = self.cell_types[i,:].type(torch.FloatTensor)
        positive_weights = self.positive_weights.type(torch.FloatTensor)
        return {"expressions": expressions, "cell_types": cell_types, "positive_weights": positive_weights}