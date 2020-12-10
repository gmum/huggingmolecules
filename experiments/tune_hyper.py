from experiments.training import tune_hyper
from src.huggingmolecules.utils import *

apply_gin_config(configs=['experiments/configs/bases/tune_hyper.gin'])

tune_hyper()
