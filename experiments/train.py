import logging
import argparse

from experiments.src import train_model
from experiments.src.gin import apply_gin_config, gin

logging.basicConfig(format='%(levelname)s:%(module)s: %(message)s', level=logging.INFO)

additional_args = {
    'name.prefix': {'type': str, 'default': 'Train'}
}

apply_gin_config(base='train', additional_args=additional_args)

train_model()
