import pytorch_lightning as pl

from src.chemformers.featurization.featurization_api import BatchEncodingProtocol
from src.chemformers.models.models_api import PretrainedModelBase


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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
