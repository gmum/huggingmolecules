import logging

from experiments.training import train_model
from src.huggingmolecules.utils import *

logging.basicConfig(format='%(levelname)s:%(module)s: %(message)s', level=logging.INFO)

apply_gin_config(configs=['experiments/configs/bases/train.gin'])

train_model()
