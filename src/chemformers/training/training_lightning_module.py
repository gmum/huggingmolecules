from typing import Generic

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from src.chemformers.featurization.featurization_api import T_BatchEncoding, BatchEncodingProtocol
from src.chemformers.models.models_api import PretrainedModelBase


class TrainingModule(pl.LightningModule):
    def __init__(self, model: PretrainedModelBase, *, loss_fn=None, optimizer=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn if loss_fn else F.mse_loss
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def forward(self, batch: BatchEncodingProtocol):
        return self.model.forward(batch)

    def training_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_loss_step', loss, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('valid_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer
