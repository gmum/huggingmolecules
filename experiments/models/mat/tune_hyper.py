from src.huggingmolecules import MatModel, MatFeaturizer
from experiments.training import tune_hyper
from src.huggingmolecules.utils import *

apply_gin_config()
model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()
tune_hyper(model, featurizer)
