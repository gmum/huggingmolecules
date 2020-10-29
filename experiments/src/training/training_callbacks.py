import logging
import os
from typing import List

import gin
from pytorch_lightning import Callback

from experiments.src.gin.gin_utils import get_formatted_config_str
from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin


class NeptuneCompatibleCallback(Callback):
    def __init__(self):
        super(NeptuneCompatibleCallback, self).__init__()
        self.neptune = None


class LRSchedulerBase(NeptuneCompatibleCallback):
    def __init__(self):
        super().__init__()
        self.base_lr = None
        self.total_steps = None

    def on_train_start(self, trainer, pl_module):
        self.base_lr = trainer.optimizers[0].param_groups[0]['lr']
        self.total_steps = len(pl_module.train_dataloader.dataloader) * trainer.max_epochs

        logging.info(f'Set base_lr to: {self.base_lr}')
        logging.info(f'Set total_steps to: {self.total_steps}')

    def get_lr(self, step):
        raise NotImplementedError

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        step = trainer.global_step
        for i, group in enumerate(trainer.optimizers[0].param_groups):
            group['lr'] = self.get_lr(step)

            logging.info(f'Set group-{i}-lr to {group["lr"]}')
            if self.neptune:
                self.neptune.log_metric(f'group-{i}-lr', group['lr'])


@gin.configurable
class NoamLRScheduler(LRSchedulerBase):
    def __init__(self, warmup_factor: int, model_size: int):
        super().__init__()
        self.warmup_factor = warmup_factor
        self.model_size = model_size
        self.warmup_steps = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.warmup_steps = self.warmup_factor * self.total_steps

    def get_lr(self, step):
        step += 1
        return self.base_lr * 100 * (self.model_size ** (-0.5) * min(step ** (-0.5),
                                                                     step * (1e-6 + self.warmup_steps) ** (-1.5)))


@gin.configurable
class EnhancedNoamLRScheduler(LRSchedulerBase):
    def __init__(self, warmup_factor: int, init_lr_ratio: float = 10, final_lr_ratio: float = 6):
        super().__init__()
        self.warmup_factor = warmup_factor
        self.init_lr_ratio = init_lr_ratio
        self.final_lr_ratio = final_lr_ratio
        self.warmup_steps = None
        self.init_lr = None
        self.final_lr = None
        self.linear_increment = None
        self.exp_gamma = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.warmup_steps = self.warmup_factor * self.total_steps
        self.init_lr = self.base_lr / self.init_lr_ratio
        self.final_lr = self.base_lr / self.final_lr_ratio
        self.linear_increment = (self.base_lr - self.init_lr) / self.warmup_steps
        self.exp_gamma = (self.final_lr / self.base_lr) ** (1 / (self.total_steps - self.warmup_steps))

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.init_lr + step * self.linear_increment
        else:
            return self.base_lr * (self.exp_gamma ** (step - self.warmup_steps))


class GinConfigSaver(NeptuneCompatibleCallback):
    def __init__(self, target_name: str = "gin-config-all.txt", excluded_namespaces: List[str] = None):
        super().__init__()
        self.target_name = target_name
        self.excluded_namespaces = excluded_namespaces

    def on_train_start(self, trainer, pl_module):
        gin_str = get_formatted_config_str(excluded=self.excluded_namespaces)
        target_path = os.path.join(trainer.default_root_dir, self.target_name)
        with open(target_path, "w") as f:
            f.write(gin_str)
        if self.neptune:
            self.neptune.log_artifact(target_path)


class ModelConfigSaver(NeptuneCompatibleCallback):
    def __init__(self, target_name: str = "model_config.json"):
        super().__init__()
        self.target_name = target_name

    def on_train_start(self, trainer, pl_module):
        config: PretrainedConfigMixin = pl_module.model.config
        target_path = os.path.join(trainer.default_root_dir, self.target_name)
        config.save(target_path)
        if self.neptune:
            self.neptune.log_artifact(target_path)
