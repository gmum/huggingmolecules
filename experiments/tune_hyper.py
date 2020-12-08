from experiments.training import tune_hyper
from experiments.training.training_train_model_utils import get_model_and_featurizer
from src.huggingmolecules.utils import *

apply_gin_config(configs=['experiments/configs/bases/tune_hyper.gin'])
model, featurizer = get_model_and_featurizer()
tune_hyper(model, featurizer)
