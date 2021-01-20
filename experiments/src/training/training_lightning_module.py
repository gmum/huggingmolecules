from typing import Optional

import pytorch_lightning as pl

from experiments.src.training.training_metrics import BatchWeightedLoss
from src.huggingmolecules.featurization.featurization_api import BatchEncodingProtocol
from src.huggingmolecules.models.models_api import PretrainedModelBase


class TrainingModule(pl.LightningModule):
    def __init__(self, model: PretrainedModelBase, *, loss_fn, optimizer, metric_cls):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.outputs = {phase: [] for phase in ['valid', 'test']}

        self.metric_loss = {phase: BatchWeightedLoss() for phase in ['train', 'valid', 'test']}
        self.metric = {phase: metric_cls() for phase in ['train', 'valid', 'test']}
        self.metric_name = metric_cls.__name__.lower()

    def cuda(self, device: Optional[int] = None):
        setattr(self.model, 'device', device)
        return super(TrainingModule, self).cuda(device)

    def forward(self, batch: BatchEncodingProtocol):
        return self.model.forward(batch)

    def _step(self, mode: str, batch: BatchEncodingProtocol, batch_idx: int):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        preds = output[0] if isinstance(output, tuple) else output

        self.metric_loss[mode](loss, len(batch))
        self.metric[mode](preds, batch.y)

        self.log(f'{mode}_loss', self.metric_loss[mode], on_epoch=True, on_step=False)
        self.log(f'{mode}_{self.metric_name}', self.metric[mode], on_epoch=True, on_step=False)

        return {'loss': loss, 'output': output}

    def training_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        return self._step('train', batch, batch_idx)['loss']

    def on_validation_epoch_start(self) -> None:
        self.outputs['valid'] = []

    def validation_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        outputs = self._step('valid', batch, batch_idx)
        self.outputs['valid'].append(outputs['output'])
        return outputs['loss']

    def on_test_epoch_start(self) -> None:
        self.outputs['test'] = []

    def test_step(self, batch: BatchEncodingProtocol, batch_idx: int):
        outputs = self._step('test', batch, batch_idx)
        self.outputs['test'].append(outputs['output'])
        return outputs['loss']

    def configure_optimizers(self):
        return self.optimizer
