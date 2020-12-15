import logging
import os
from typing import List

import gin
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from experiments.src.gin import get_formatted_config_str


class NeptuneCompatibleCallback(Callback):
    def __init__(self):
        super(NeptuneCompatibleCallback, self).__init__()
        self.neptune = None


class LRSchedulerBase(NeptuneCompatibleCallback):
    def __init__(self, warmup_factor: int):
        super().__init__()
        self.warmup_factor = warmup_factor
        self.base_lr = None
        self.warmup_steps = None
        self.total_steps = None

    def on_train_start(self, trainer, pl_module):
        self.base_lr = trainer.optimizers[0].param_groups[0]['lr']
        self.total_steps = len(pl_module.train_dataloader.dataloader) * trainer.max_epochs
        self.warmup_steps = self.warmup_factor * self.total_steps

        logging.info(f'Set base_lr to: {self.base_lr}')
        logging.info(f'Set warmup_steps to: {self.warmup_steps}')
        logging.info(f'Set total_steps to: {self.total_steps}')

    def get_multiplier(self, step):
        raise NotImplementedError

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        step = trainer.global_step + 1
        for i, group in enumerate(trainer.optimizers[0].param_groups):
            group['lr'] = self.base_lr * self.get_multiplier(step)

            logging.info(f'Set group-{i}-lr to {group["lr"]}')
            if self.neptune:
                self.neptune.log_metric(f'group-{i}-lr', group['lr'])


@gin.configurable
class NoamLRScheduler(LRSchedulerBase):
    def __init__(self, model_size_name: str, warmup_factor: int):
        super().__init__(warmup_factor)
        self.model_size_name = model_size_name
        self.model_size = None

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        config = pl_module.model.get_config()
        self.model_size = getattr(config, self.model_size_name)

        logging.info(f'Set model_size to: {self.model_size}')

    def get_multiplier(self, step):
        return 100 * (self.model_size ** (-0.5) * min(step ** (-0.5),
                                                      step * (1e-6 + self.warmup_steps) ** (-1.5)))


@gin.configurable
class LinearLRScheduler(LRSchedulerBase):
    def __init__(self, warmup_factor: int):
        super().__init__(warmup_factor)

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)

    def get_multiplier(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0, float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps))
        )


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
        config = pl_module.model.get_config()
        target_path = os.path.join(trainer.default_root_dir, self.target_name)
        config.save(target_path)
        if self.neptune:
            self.neptune.log_artifact(target_path)


@gin.configurable('ModelCheckpoint', blacklist=['filepath'])
class ConfigurableModelCheckpoint(ModelCheckpoint):
    def __init__(self, *,
                 filepath,
                 verbose=True,
                 save_last=True,
                 monitor='valid_loss',
                 mode='min', **kwargs):
        super().__init__(filepath=filepath,
                         verbose=verbose,
                         save_last=save_last,
                         monitor=monitor,
                         mode=mode,
                         **kwargs)
