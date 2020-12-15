from experiments.src import tune_hyper
from experiments.src.gin import apply_gin_config

apply_gin_config(base='tune_hyper')

tune_hyper()
