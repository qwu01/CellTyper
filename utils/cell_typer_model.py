import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from pathlib import Path
import pandas as pd
from scipy.sparse import load_npz
from scipy.io import mmread as readMM
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LinearSVD(nn.Module):
    """SVD implementation taken from Deep.Bayes Summer school
    """
    def __init__(self, in_features, out_features, bias=True):
        super(LinearSVD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)        
        
    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10) 
        
        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
    
        return F.linear(x, self.W * (self.log_alpha < 3).float()) + self.bias

    def compute_kl(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        kl = k1*torch.sigmoid(k2 + k3*self.log_alpha)-0.5*torch.log1p(torch.exp(-self.log_alpha))
        kl = -torch.sum(kl)
        return kl

class CellTyper(pl.LightningModule):
    def __init__(self, args):
        super(CellTyper, self).__init__()
        self.save_hyperparameters()
        if self.dropout_type == "variational":
            self.initiate_variational()
        elif self.dropout_type == "standard":
            self.initiate_standard()

        self.get_dummies = OneHotEncoder(handle_unknown="ignore")
        self.standard_scaler = StandardScaler()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--num_hidden_layers", type=int, default=0)
        parser.add_argument("--hidden_size", type=int, default=None)

        parser.add_argument("--dropout_type", type=str, default="variational", choices=("variational", "standard"))
        parser.add_argument("--kl_weight", type=float, default=0.04)
        parser.add_argument("--dropout_rate", type=float, default=0.95) # input dropout
        parser.add_argument("--hidden_dropout_rate", type=float, default=0.20)

        parser.add_argument("--training_set_folder", type=str, default="Data/datasets/pbmc1_10x Chromium (v2) A_training/") # for later using different set of training data
        parser.add_argument("--validation_set_folder", type=str, default="Data/datasets/pbmc1_10x Chromium (v2) A_validation") # for later using different set of training data
        parser.add_argument("--test_set_folder", type=str, default="Data/datasets/pbmc1_10x Chromium (v2) A_test") # for later using different set of training data
        parser.add_argument("--batch_size", type=int, default=100)

        return parser

    def initiate_variational(self):
        if self.hidden_size == 0:
            self.linear = LinearSVD(self.num_genes, self.num_labels)
        elif self.hidden_size > 0:
            self.input_layer = LinearSVD(self.num_genes, self.hidden_size)
            self.non_linear = nn.ReLU()
            self.output_layer = LinearSVD(self.hidden_size, self.num_labels)
            self.hidden_layers = nn.ModuleList([LinearSVD(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers)])

    def initiate_standard(self):
        if self.hidden_size == 0:
            self.linear = nn.Linear(self.num_genes, self.num_labels)
            self.dropout = nn.Dropout(self.dropout_rate)
        elif self.hidden_size > 0:
            self.input_layer = nn.Linear(self.num_genes, self.hidden_size)
            self.non_linear = nn.ReLU()
            self.dropout = nn.Dropout(self.dropout_rate)
            self.output_layer = nn.Linear(self.hidden_size, self.num_labels)
            self.hidden_dropout = nn.Dropout(self.hidden_dropout_rate)
            self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers)])

    def forward_variational(self, x):
        if self.hidden_size == 0:
            x = self.linear(x)
        elif self.hidden_size > 0:
            x = self.input_layer(x)
            x = self.non_linear(x)
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
                x = self.non_linear(x)
            x = self.output_layer(x)
        return x
    
    def forward_standard(self, x):
        if self.hidden_size == 0:
            x = self.dropout(x)
            x = self.linear(x)
        elif self.hidden_size > 0:
            x = self.dropout(x)
            x = self.input_layer(x)
            x = self.non_linear(x)
            x = self.hidden_dropout(x)
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
                x = self.non_linear(x)
                x = self.hidden_dropout(x)
            x = self.output_layer(x)

    def forward(self, x):
        if self.dropout_type == "variational":
            x = self.forward_variational(x)
        elif self.dropout_type == "standard":
            x = self.forward_standard(x)
        return x


    def shared_step(self, batch, pos_weights):
        expressions = batch['expressions']
        labels = batch['cell_types'] # (cell x labels)
        predictions = self(expressions)
        if self.dropout_type == "variational":
            self.kl_weight = min(self.kl_weight+0.02, 1)
            kl = 0.0
            for m in self.children():
                if hasattr(m, 'compute_kl'):
                    kl = kl + m.compute_kl()
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels)*self.training_set_size + self.kl_weight*kl
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels)
        return loss, predictions, labels

    def training_step(self, batch, i):
        pos_weights = self.position_weights["training"]
        loss, predictions, labels = self.shared_step(batch, pos_weights)
        train_acc = ((predictions>0).int() == labels.int().to(self.device)).sum()/labels.numel()
        self.log('training_loss', loss, prog_bar=True)
        self.log('training_accuracy', train_acc, prog_bar=True)
        return {'loss': loss, 'accuracy': train_acc}

    def validation_step(self, batch, i):
        pos_weights = self.position_weights["validation"]
        loss, predictions, labels = self.shared_step(batch, pos_weights)
        val_acc = ((predictions>0).int() == labels.int().to(self.device)).sum()/labels.numel()
        return {"val_loss": loss, "val_accuracy": val_acc, "predictions": predictions, "labels": labels}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.log('avg_accuracy', avg_acc)
        val_predictions = torch.stack([x['predictions'] for x in outputs])
        val_labels = torch.stack([x['labels'] for x in outputs])
        # do something (e.g. ROC/Pr-Recall) #NOTE: need `validation_step` to return all predictions and labels

    def test_step(self, batch, i):
        pos_weights = self.position_weights["test"]
        loss, predictions, labels = self.shared_step(batch, pos_weights)
        test_acc = ((predictions>0).int() == labels.int().to(self.device)).sum()/labels.numel()
        return {"test_loss": loss, "test_accuracy": test_acc, "predictions": predictions, "labels": labels}

    def test_epoch_end(self, outputs):
        all_test_step_outs = outputs.out
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        self.log('average_test_loss', avg_loss)
        self.log('average_test_acc', avg_acc)
        val_predictions = torch.stack([x['predictions'] for x in outputs])
        val_labels = torch.stack([x['labels'] for x in outputs])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def setup(self):
        #TODO try torch.sparse
        folders = {
            "training": Path(self.training_set_folder),
            "validation": Path(self.validation_set_folder),
            "test": Path(self.test_set_folder)
        } # Starting from Python 3.7, insertion order of Python dictionaries is guaranteed.

        genes = pd.read_csv(folders["training"]/"gene.csv", index_col=0)
        for key, path in folders.items(): # ['training', 'validation', 'test]
            cell = pd.read_csv(path/"cell.csv", index_col=0)
            cell = cell.reset_index(drop=True)[['Name', 'CellType']]
            if key == "training":
                self.get_dummies.fit(cell[["CellType"]])
            cell_type = self.get_dummies.transform(cell[['CellType']]).toarray()
            cell_type = torch.Tensor(cell_type).type(torch.FloatTensor) # label tensor
            positive_weight = (cell_type.shape[0] - cell_type.sum(axis=0))/cell_type.sum(axis=0) # pos weight
            expression = readMM(path/"log_norm_count_sparse.mtx")
            expression = expression.tocsr()
            expression = torch.transpose(torch.tensor(expression.todense()).type(torch.FloatTensor), 0, 1) # T(expression) (cell x gene)
            if key == "training":
                # scale the dataset. Save the scale parameters, use the parameters to scale validation/test dataset.
                self.standard_scaler.fit(expression)
                expression = torch.tensor(self.standard_scaler.transform(expression))
                self.examples_training = CellTyperDataSet(cell_type, expression)
            elif key == "validation":
                expression = torch.tensor(self.standard_scaler.transform(expression))
                self.examples_validation = CellTyperDataSet(cell_type, expression)
            elif key == "test":
                expression = torch.tensor(self.standard_scaler.transform(expression))
                self.examples_test = CellTyperDataSet(cell_type, expression)

    def train_dataloader(self):
        training_data_loader = DataLoader(dataset=self.examples_training, batch_size=self.batch_size)
        return training_data_loader
    
    def val_dataloader(self):
        val_data_loader = DataLoader(dataset=self.examples_validation, batch_size=self.batch_size)
        return val_data_loader
    
    def test_dataloader(self):
        test_data_loader = DataLoader(dataset=self.examples_test, batch_size=self.batch_size)
        return test_data_loader


class CellTyperDataSet(Dataset):

    # training set size
    # number of features
    # number of classes (labels)
    def __init__(self, cell_types, expressions):
        self.cell_types = cell_types
        self.expressions = expressions
        self.expressions = torch.tensor(expressions.todense()).type(torch.FloatTensor)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, i) -> torch.Tensor:
        
        expressions = torch.tensor(self.expressions[:,i].todense()).type(torch.FloatTensor)
        cell_types = torch.tensor(self.cell_types[i,:].todense()).type(torch.FloatTensor)
        
        return {
            "expressions": expressions,
            "cell_types": cell_types,
        }