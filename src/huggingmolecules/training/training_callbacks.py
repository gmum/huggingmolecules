import logging
import os
from typing import List

import gin
from pytorch_lightning import Callback

from src.huggingmolecules.utils import get_formatted_config_str, parse_gin_str


class NeptuneCompatibleCallback(Callback):
    def __init__(self):
        super(NeptuneCompatibleCallback, self).__init__()
        self.neptune = None


@gin.configurable
class NoamLRScheduler(NeptuneCompatibleCallback):
    def __init__(self, model_size_name: str, warmup_factor: int):
        super().__init__()
        self.model_size_name = model_size_name
        self.warmup_factor = warmup_factor
        self.factor = None
        self.warmup_steps = None
        self.model_size = None

    def on_train_start(self, trainer, pl_module):
        self.factor = 100 * trainer.optimizers[0].param_groups[0]['lr']
        total_steps = len(pl_module.train_dataloader.dataloader) * trainer.max_epochs
        self.warmup_steps = self.warmup_factor * total_steps
        config = pl_module.model.get_config()
        self.model_size = getattr(config, self.model_size_name)

        logging.info(f'Set warmup_steps to: {self.warmup_steps}')
        logging.info(f'Set model_size to: {self.model_size}')
        logging.info(f'Set factor to: {self.factor}')

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        step = trainer.global_step + 1
        for i, group in enumerate(trainer.optimizers[0].param_groups):
            group['lr'] = self.factor * \
                          (self.model_size ** (-0.5) * min(step ** (-0.5),
                                                           step * (1e-6 + self.warmup_steps) ** (-1.5)))
            logging.info(f'Set group-{i}-lr to {group["lr"]}')
            if self.neptune:
                self.neptune.log_metric(f'group-{i}-lr', group['lr'])


class HyperparamsNeptuneSaver(NeptuneCompatibleCallback):
    def on_train_start(self, trainer, pl_module):
        gin_str = get_formatted_config_str(excluded=['neptune', 'optuna', 'macro'])
        parsed_gin_str = parse_gin_str(gin_str)
        self.neptune.log_hyperparams(parsed_gin_str)


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
