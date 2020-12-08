import logging

from experiments.training import train_model
from experiments.training.training_train_model_utils import get_model_and_featurizer
from src.huggingmolecules.utils import *

# logging.basicConfig(format='%(levelname)s:%(module)s: %(message)s', level=logging.INFO)
apply_gin_config(configs=['experiments/configs/bases/train.gin'])
model, featurizer = get_model_and_featurizer()
train_model(model, featurizer)
