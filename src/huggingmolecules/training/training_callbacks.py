import datetime
import json
import logging
import os
import sys
import time

import gin
from pytorch_lightning import Callback


class NeptuneCompatibleCallback(Callback):
    def __init__(self):
        super(NeptuneCompatibleCallback, self).__init__()
        self.neptune = None


@gin.configurable()
class LRScheduler(NeptuneCompatibleCallback):
    def __init__(self, model_size_name: str, warmup_factor: int):
        super(LRScheduler, self).__init__()
        self.model_size_name = model_size_name
        self.warmup_factor = warmup_factor

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


class MetaSaver(Callback):
    def __init__(self):
        super(MetaSaver, self).__init__()

    def on_train_start(self, trainer, pl_module):
        logging.info("Saving meta data information from the beginning of training")

        assert os.system(
            "cp {} {}".format(sys.argv[0], trainer.default_root_dir)) == 0, "Failed to execute cp of source script"

        utc_date = datetime.datetime.utcnow().strftime("%Y_%m_%d")

        time_start = time.time()
        cmd = "python " + " ".join(sys.argv)
        self.meta = {"cmd": cmd,
                     "save_path": trainer.default_root_dir,
                     "most_recent_train_start_date": utc_date,
                     "execution_time": -time_start}

        json.dump(self.meta, open(os.path.join(trainer.default_root_dir, "meta.json"), "w"), indent=4)

    def on_train_end(self, trainer, pl_module):
        self.meta['execution_time'] += time.time()
        json.dump(self.meta, open(os.path.join(trainer.default_root_dir, "meta.json"), "w"), indent=4)
        os.system("touch " + os.path.join(trainer.default_root_dir, "FINISHED"))


class Heartbeat(Callback):
    def __init__(self, interval=10):
        self.last_time = time.time()
        self.interval = interval

    def on_train_start(self, trainer, pl_module):
        logging.info("HEARTBEAT - train begin")
        os.system("touch " + os.path.join(trainer.default_root_dir, "HEARTBEAT"))

    def on_batch_start(self, trainer, pl_module):
        if time.time() - self.last_time > self.interval:
            logging.info("HEARTBEAT")
            os.system("touch " + os.path.join(trainer.default_root_dir, "HEARTBEAT"))
            self.last_time = time.time()
