from src.huggingmolecules import MatModel, MatFeaturizer, MatConfig
from src.huggingmolecules.training import train_model
from src.huggingmolecules.utils import *

apply_gin_config()
model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()
train_model(model, featurizer)
