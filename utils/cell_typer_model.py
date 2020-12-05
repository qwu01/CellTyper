import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

class LinearSVD(nn.Module):
    """SVD implementation taken from Deep.Bayes Summer school
    """
    def __init__(self, in_features, out_features):
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
    def __init__(self, args, training_set_size, num_labels, num_genes):
        super(CellTyper, self).__init__()
        self.training_set_size = training_set_size
        self.num_labels = num_labels
        self.num_genes = num_genes
        self.save_hyperparameters(args)

        if self.hparams.dropout_type == "variational":
            self.initiate_variational()
        elif self.hparams.dropout_type == "standard":
            self.initiate_standard()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--num_hidden_layers", type=int, default=0)
        parser.add_argument("--hidden_size", type=int, default=256)

        parser.add_argument("--dropout_type", type=str, default="variational", choices=("variational", "standard"))
        parser.add_argument("--kl_weight", type=float, default=0.04)
        parser.add_argument("--dropout_rate", type=float, default=0.95) # input dropout
        parser.add_argument("--hidden_dropout_rate", type=float, default=0.20)

        parser.add_argument("--learning_rate", type=float, default=1e-3)

        return parser

    def initiate_variational(self):
        if self.hparams.num_hidden_layers == 0:
            self.linear = LinearSVD(self.num_genes, self.num_labels)
        elif self.hparams.num_hidden_layers > 0:
            self.input_layer = LinearSVD(self.num_genes, self.hparams.hidden_size)
            self.non_linear = nn.ReLU()
            self.output_layer = LinearSVD(self.hparams.hidden_size, self.num_labels)
            self.hidden_layers = nn.ModuleList([LinearSVD(self.hparams.hidden_size, self.hparams.hidden_size) for _ in range(self.hparams.num_hidden_layers)])

    def initiate_standard(self):
        if self.hparams.num_hidden_layers == 0:
            self.linear = nn.Linear(self.num_genes, self.num_labels)
            self.dropout = nn.Dropout(self.hparams.dropout_rate)
        elif self.hparams.num_hidden_layers > 0:
            self.input_layer = nn.Linear(self.num_genes, self.hparams.hidden_size)
            self.non_linear = nn.ReLU()
            self.dropout = nn.Dropout(self.hparams.dropout_rate)
            self.output_layer = nn.Linear(self.hparams.hidden_size, self.num_labels)
            self.hidden_dropout = nn.Dropout(self.hparams.hidden_dropout_rate)
            self.hidden_layers = nn.ModuleList([nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size) for _ in range(self.hparams.num_hidden_layers)])

    def forward_variational(self, x):
        if self.hparams.num_hidden_layers == 0:
            x = self.linear(x)
        elif self.hparams.num_hidden_layers > 0:
            x = self.input_layer(x)
            x = self.non_linear(x)
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
                x = self.non_linear(x)
            x = self.output_layer(x)
        return x
    
    def forward_standard(self, x):
        if self.hparams.num_hidden_layers == 0:
            x = self.dropout(x)
            x = self.linear(x)
        elif self.hparams.num_hidden_layers > 0:
            x = self.dropout(x)
            x = self.input_layer(x)
            x = self.non_linear(x)
            x = self.hidden_dropout(x)
            for hidden_layer in self.hidden_layers:
                x = hidden_layer(x)
                x = self.non_linear(x)
                x = self.hidden_dropout(x)
            x = self.output_layer(x)
        return x

    def forward(self, x):
        if self.hparams.dropout_type == "variational":
            x = self.forward_variational(x)
        elif self.hparams.dropout_type == "standard":
            x = self.forward_standard(x)
        return x

    def shared_step(self, batch):
        pos_weights = batch['positive_weights']
        expressions = batch['expressions'] # (cell x gene)

        labels = batch['cell_types'] # (cell x labels)
        predictions = self(expressions)
        if self.hparams.dropout_type == "variational":
            self.hparams.kl_weight = min(self.hparams.kl_weight+0.02, 1)
            kl = 0.0
            for m in self.children():
                if hasattr(m, 'compute_kl'):
                    kl = kl + m.compute_kl()
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels)*self.training_set_size + self.hparams.kl_weight*kl
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels)
        return loss, predictions, labels

    def training_step(self, batch, i):
        loss, predictions, labels = self.shared_step(batch)
        train_acc = ((predictions>0).int() == labels.int().to(self.device)).sum()/labels.numel()
        self.log('training_loss', loss, prog_bar=True)
        self.log('training_accuracy', train_acc, prog_bar=True)
        return {"loss": loss, "accuracy": train_acc, "predictions": predictions, "labels": labels}

    def validation_step(self, batch, i):
        loss, predictions, labels = self.shared_step(batch)
        val_acc = ((predictions>0).int() == labels.int().to(self.device)).sum()/labels.numel()
        return {"val_loss": loss, "val_accuracy": val_acc, "predictions": predictions, "labels": labels}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.cat([x['val_loss'].unsqueeze(0) for x in outputs]).mean()
        avg_acc = torch.cat([x['val_accuracy'].unsqueeze(0) for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_accuracy', avg_acc)
        val_predictions = torch.cat([x['predictions'] for x in outputs])
        val_labels = torch.cat([x['labels'] for x in outputs])
        print()
        self.log('AUC_ROC', roc_auc_score(y_true=val_labels.cpu(), y_score=val_predictions.cpu()))
        self.log('F1_score', f1_score(y_true=val_labels.cpu(), y_pred=(val_predictions>0).cpu(), average="samples"))
        self.log('Average_precision_recall (AP)', average_precision_score(y_true=val_labels.cpu(), y_score=val_predictions.cpu(), average="samples"))

    def test_step(self, batch, i):
        loss, predictions, labels = self.shared_step(batch)
        test_acc = ((predictions>0).int() == labels.int().to(self.device)).sum()/labels.numel()
        return {"test_loss": loss, "test_accuracy": test_acc, "predictions": predictions, "labels": labels}

    def test_epoch_end(self, outputs):
        avg_loss = torch.cat([x['test_loss'].unsqueeze(0) for x in outputs]).mean()
        avg_acc = torch.cat([x['test_accuracy'].unsqueeze(0) for x in outputs]).mean()
        self.log('average_test_loss', avg_loss)
        self.log('average_test_acc', avg_acc)
        test_predictions = torch.cat([x['predictions'] for x in outputs])
        test_labels = torch.cat([x['labels'] for x in outputs])
        self.log('AUC_ROC', roc_auc_score(y_true=test_labels.cpu(), y_score=test_predictions.cpu()))
        self.log('F1_score', f1_score(y_true=test_labels.cpu(), y_pred=(test_predictions>0).cpu(), average="samples"))
        self.log('Average_precision_recall (AP)', average_precision_score(y_true=test_labels.cpu(), y_score=test_predictions.cpu(), average="samples"))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = MultiStepLR(optimizer, milestones=[100,200,500,2000,4000,8000], gamma=0.2)
        return [optimizer], [scheduler]
