import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CellTyperClsHead(nn.Module):

    def __init__(self, hidden_size, cls_size):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, cls_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(cls_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class CellTyper(pl.LightningModule):

    def __init__(self, args):

        super(CellTyper, self).__init__()
        self.args = args
        self._parse_args()

        self.linear = nn.Linear(self.feature_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm((1, self.hidden_size), elementwise_affine=False) # without Learnable Parameters
        self.non_linear = nn.ReLU()
        self.dropout90 = nn.Dropout(0.9)
        if self.num_hidden_layers > 1:
            self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hidden_layers-1)])
        self.cls_head = CellTyperClsHead(self.hidden_size, self.cls_size)

    def _parse_args(self):
        self.num_hidden_layers = self.args.num_hidden_layers
        self.hidden_size = self.args.hidden_size
        self.cls_size = self.args.cls_size
        self.feature_size = self.args.feature_size

        self.learning_rate = self.args.learning_rate

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.non_linear(x)
        x = self.dropout90(x)
        if self.num_hidden_layers > 1:
            for Linear in self.linears:
                x = Linear(x)
                x = self.layer_norm(x)
                x = self.non_linear(x)
                x = self.dropout90(x)
        x = self.cls_head(x)
        return x

    def training_step(self, batch, i):
        pos_weights = batch['positives_weights']
        expressions = batch['expressions'] # (gene x cell, need transpose)
        labels = batch['cell_types'] # (cell x labels)
        predictions = self(expressions)
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels)
        self.log('training_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, i):
        pos_weights = batch['positives_weights']
        expressions = batch['expressions'] # (gene x cell, need transpose)
        labels = batch['cell_types'] # (cell x labels)
        predictions = self(expressions)
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)(predictions, labels)
        val_acc = ((predictions>0).int() == labels.int().cuda()).sum()/labels.numel()
        return {'val_loss': loss, 'val_accuracy': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        self.log('average_val_loss', avg_loss)
        self.log('average_val_acc', avg_acc)
        return {'average_val_loss': avg_loss, 'average_val_acc': avg_acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)