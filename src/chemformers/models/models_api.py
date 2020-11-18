from typing import Generic, TypeVar

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.chemformers.featurization.featurization_api import T_BatchEncoding

T_Config = TypeVar("T_Config")


class LightningModuleMixin(pl.LightningModule, Generic[T_BatchEncoding]):
    def forward(self, batch: T_BatchEncoding):
        raise NotImplementedError

    def training_step(self, batch: T_BatchEncoding, batch_idx: int):
        output = self.forward(batch)
        loss = F.mse_loss(output, batch.y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_loss_step', loss, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch: T_BatchEncoding, batch_idx: int):
        output = self.forward(batch)
        loss = F.mse_loss(output, batch.y)
        self.log('valid_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch: T_BatchEncoding, batch_idx: int):
        output = self.forward(batch)
        loss = F.mse_loss(output, batch.y)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class PretrainedModelMixin:
    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def _get_config_cls(cls):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        file_path = cls._get_arch_from_pretrained_name(pretrained_name)
        config_cls = cls._get_config_cls()
        config = config_cls.from_pretrained(pretrained_name)
        model = cls(config)
        model.load_weights(file_path)
        return model

    def load_weights(self, file_path: str):
        pretrained_state_dict = torch.load(file_path)
        if 'model' in pretrained_state_dict:
            pretrained_state_dict = pretrained_state_dict['model']
        model_state_dict = self.state_dict()
        for name, param in pretrained_state_dict.items():
            if 'generator' in name:
                continue
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            model_state_dict[name].copy_(param)

    def save_weights(self):
        pass
