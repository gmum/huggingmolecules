import argparse

from experiments.src import tune_hyper
from experiments.src.benchmarking.benchmarking_compute_results import compute_benchmark_results
from experiments.src.gin import apply_gin_config

parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')

args = apply_gin_config(base='benchmark', parser=parser)

if not args.results_only:
    tune_hyper()
else:
    compute_benchmark_results()
