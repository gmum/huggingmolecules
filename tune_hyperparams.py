from src.chemformers import MatModel, MatFeaturizer
from src.chemformers.training import train_model, tune_hyperparams
from src.chemformers.utils import *

apply_gin_config()
model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()
tune_hyperparams(model, featurizer)
