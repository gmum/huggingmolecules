from typing import Generic

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from src.chemformers.featurization.featurization_api import T_BatchEncoding


# T_Config_Cls = TypeVar("T_Config_Cls", bound=Type[PretrainedConfigMixin])
# T_Model = TypeVar("T_Model")


class PretrainedModelBase(pl.LightningModule, Generic[T_BatchEncoding]):
    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def _get_config_cls(cls):
        raise NotImplementedError

    def forward(self, batch: T_BatchEncoding):
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

    # Pytorch lightning

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = F.mse_loss(output, batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = F.mse_loss(output, batch.y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = F.mse_loss(output, batch.y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
