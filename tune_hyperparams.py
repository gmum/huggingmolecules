from src.huggingmolecules import MatModel, MatFeaturizer
from src.huggingmolecules.training import train_model, tune_hyperparams
from src.huggingmolecules.utils import *

apply_gin_config()
model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()
tune_hyperparams(model, featurizer)
