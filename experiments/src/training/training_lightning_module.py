from typing import Optional

import pytorch_lightning as pl
import torch.nn.functional as F

from experiments.src.training.training_metrics import AUROC
from src.huggingmolecules.featurization.featurization_api import BatchEncodingProtocol
from src.huggingmolecules.models.models_api import PretrainedModelBase


class TrainingModule(pl.LightningModule):
    def __init__(self, model: PretrainedModelBase, *, loss_fn, optimizer, task: str):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.task = task

        if task == 'classification':
            self.auroc = {phase: AUROC() for phase in ['train', 'valid', 'test']}

    def cuda(self, device: Optional[int] = None):
        setattr(self.model, 'device', device)
        return super(TrainingModule, self).cuda(device)

    def forward(self, batch: BatchEncodingProtocol):
        return self.model.forward(batch)

    def _step(self, phase: str, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log(f'{phase}_loss', loss, on_epoch=True)

        if self.task == 'classification':
            logits = F.sigmoid(output)
            self.auroc[phase](logits, batch.y)
            self.log(f'{phase}_auroc', self.auroc[phase], on_epoch=True, on_step=False)

        return loss, output

    def training_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        loss, _ = self._step('train', batch, batch_idx)
        return loss

    def validation_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        loss, _ = self._step('valid', batch, batch_idx)
        return loss

    def test_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        loss, _ = self._step('test', batch, batch_idx)
        return loss

    def configure_optimizers(self):
        return self.optimizer
