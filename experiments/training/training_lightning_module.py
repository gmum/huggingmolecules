import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import LambdaLR

from src.huggingmolecules.featurization.featurization_api import BatchEncodingProtocol
from src.huggingmolecules.models.models_api import PretrainedModelBase


class TrainingModule(pl.LightningModule):
    def __init__(self, model: PretrainedModelBase, *, loss_fn, optimizer):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, batch: BatchEncodingProtocol):
        return self.model.forward(batch)

    def training_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_loss_step', loss, on_epoch=False)
        return loss

    def validation_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('valid_loss', loss)
        self.log('valid_shift', torch.sum(output - batch.y))
        return loss

    def test_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('test_loss', loss)
        self.log('test_shift', torch.sum(output - batch.y))
        return loss

    def configure_optimizers(self):
        return self.optimizer
