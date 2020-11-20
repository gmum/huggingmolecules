from src.chemformers import MatModel, MatFeaturizer
from src.chemformers.training import train_model
from src.chemformers.utils import *

apply_gin_config()
model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()
train_model(model, featurizer)
