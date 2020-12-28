import argparse

from experiments.src import tune_hyper
from experiments.src.benchmarking.benchmarking_compute_results import compute_benchmark_results
from experiments.src.gin import apply_gin_config

parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')

additional_args = {
    'name.prefix': {'type': str, 'default': ''},
    'benchmark.ensemble_max_size': {'type': eval, 'default': 1},
    'benchmark.ensemble_pick_method': {'type': str, 'default': 'brute'},
    'benchmark.prefix_list': {'type': str, 'nargs': '+', 'default': None},
    'benchmark.models_names_list': {'type': str, 'nargs': '+', 'default': None}
}

args = apply_gin_config(base='benchmark', parser=parser, additional_args=additional_args)

if not args.results_only:
    tune_hyper()
else:
    compute_benchmark_results()
