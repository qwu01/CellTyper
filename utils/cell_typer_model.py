import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .layers import LinearSVD

class CellTyperClsHead(nn.Module):
    def __init__(self, hidden_size, cls_size):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, cls_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cls_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class CellTyperSVDClsHead(nn.Module):
    def __init__(self, hidden_size, cls_size):
        super().__init__()
        self.decoder = LinearSVD(hidden_size, cls_size)

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class CellTyper(pl.LightningModule):
    def __init__(self, args):
        super(CellTyper, self).__init__()
        self.args = args
        self._parse_args()

        if self.dropout_type == "variational": # use variational dropout
            self.linear = LinearSVD(self.feature_size, self.hidden_size) # input
            self.non_linear = nn.ReLU()
            self.cls_head = CellTyperSVDClsHead(self.hidden_size, self.cls_size) # out
        elif self.dropout_type == "standard":
            self.linear = nn.Linear(self.feature_size, self.hidden_size) # input
            self.layer_norm = nn.LayerNorm((1, self.hidden_size), elementwise_affine=False) # without Learnable Parameters
            self.non_linear = nn.ReLU()
            self.dropout = nn.Dropout(self.dropout_rate)
            self.cls_head = CellTyperClsHead(self.hidden_size, self.cls_size) # out
        else:
            raise Exception("Choose `dropout_type` in `[variational, standard]`")

        if self.num_hidden_layers > 0: # try not use these hidden layers. Won't need them anyway, even using MLP
            if self.dropout_type == "variational":
                self.linears = nn.ModuleList([LinearSVD(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers)])
            elif self.dropout_type == "standard":
                self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers)])

    def _parse_args(self):
        self.num_hidden_layers = self.args.num_hidden_layers
        self.hidden_size = self.args.hidden_size
        self.cls_size = self.args.cls_size
        self.feature_size = self.args.feature_size
        self.learning_rate = self.args.learning_rate
        self.dropout_type = self.args.dropout_type
        self.kl_weight = self.args.kl_weight
        self.dropout_rate = self.args.dropout_rate
        self.training_set_size = self.args.training_set_size

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        if self.dropout_type == "variational":
            x = self.non_linear(x)
        elif self.dropout_type == "standard":
            x = self.layer_norm(x)
            x = self.non_linear(x)
            x = self.dropout(x)

        if self.num_hidden_layers > 0:
            if self.dropout_type == "variational":
                for Linear in self.linears:
                    x = Linear(x)
                    x = self.non_linear(x)
            elif self.dropout_type == "standard":
                for Linear in self.linears:
                    x = Linear(x)
                    x = self.layer_norm(x)
                    x = self.non_linear(x)
                    x = self.dropout(x)
        x = self.cls_head(x)
        return x

    def training_step(self, batch, i):
        pos_weights = batch['positives_weights']
        expressions = batch['expressions'] # (gene x cell, need transpose)
        labels = batch['cell_types'] # (cell x labels)
        predictions = self(expressions)
        if self.dropout_type == "variational":
            self.kl_weight = min(self.kl_weight+0.02, 1)
            kl = 0.0
            k1, k2, k3 = 0.63576, 1.8732, 1.48695
            for m in self.children():
                if hasattr(m, '__tag__'):
                    kl = kl-torch.sum(k1*torch.sigmoid(k2+k3*m.log_alpha)-0.5*torch.log1p(torch.exp(-m.log_alpha)))
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels) * self.training_set_size + self.kl_weight * kl
            # loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels) + self.kl_weight * kl
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels)
        # train_acc = ((predictions>0).int() == labels.int().cuda()).sum()/labels.numel()
        train_acc = ((predictions>0).int() == labels.int().to(self.device)).sum()/labels.numel()
        self.log('training_loss', loss, prog_bar=True)
        self.log('training_accuracy', train_acc, prog_bar=True)
        return {'loss': loss, 'accuracy': train_acc}

    def validation_step(self, batch, i):
        pos_weights = batch['positives_weights']
        expressions = batch['expressions'] # (gene x cell, need transpose)
        labels = batch['cell_types'] # (cell x labels)
        predictions = self(expressions)
        if self.dropout_type == "variational":
            kl = 0.0
            k1, k2, k3 = 0.63576, 1.8732, 1.48695
            for m in self.children():
                if hasattr(m, '__tag__'):
                    kl = kl-torch.sum(k1*torch.sigmoid(k2+k3*m.log_alpha)-0.5*torch.log1p(torch.exp(-m.log_alpha)))
            
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels) * self.training_set_size + self.kl_weight * kl
        else:
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels)
        # val_acc = ((predictions>0).int() == labels.int().cuda()).sum()/labels.numel()
        val_acc = ((predictions>0).int() == labels.int().to(self.device)).sum()/labels.numel()
        return {'val_loss': loss, 'val_accuracy': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        self.log('average_val_loss', avg_loss)
        self.log('average_val_acc', avg_acc)
        return {'average_val_loss': avg_loss, 'average_val_acc': avg_acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

