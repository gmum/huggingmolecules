import logging

from experiments.src import train_model
from experiments.src.gin import apply_gin_config

logging.basicConfig(format='%(levelname)s:%(module)s: %(message)s', level=logging.INFO)

apply_gin_config(base='train')

train_model()
