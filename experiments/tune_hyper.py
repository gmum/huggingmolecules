from experiments.src import tune_hyper
from experiments.src.gin import apply_gin_config

additional_args = {'name.prefix': {'type': str, 'default': 'Tune'}}

apply_gin_config(base='tune_hyper', additional_args=additional_args)

tune_hyper()
