from experiments.src import tune_hyper
from experiments.src.gin import apply_gin_config

apply_gin_config(configs=['experiments/configs/bases/tune_hyper.gin'])

tune_hyper()
