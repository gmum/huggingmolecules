import logging

from src.huggingmolecules import MatModel, MatFeaturizer
from src.huggingmolecules.training import train_model
from src.huggingmolecules.utils import *

logging.basicConfig(format='%(levelname)s:%(module)s: %(message)s', level=logging.INFO)
apply_gin_config()
model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()
train_model(model, featurizer)
