import gin
from pytorch_lightning import Callback
import logging
import os
import sys
import datetime
import time
import json

logger = logging.getLogger(__name__)


@gin.configurable
class MetaSaver(Callback):
    def __init__(self):
        super(MetaSaver, self).__init__()

    def on_train_start(self, trainer, pl_module):
        logger.info("Saving meta data information from the beginning of training")

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
        logger.info("HEARTBEAT - train begin")
        os.system("touch " + os.path.join(trainer.default_root_dir, "HEARTBEAT"))

    def on_batch_start(self, trainer, pl_module):
        if time.time() - self.last_time > self.interval:
            logger.info("HEARTBEAT")
            os.system("touch " + os.path.join(trainer.default_root_dir, "HEARTBEAT"))
            self.last_time = time.time()
