import argparse

from experiments.training.training_benchmark import benchmark
from src.huggingmolecules.utils import apply_gin_config

parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')
parser.add_argument('--prefix', type=str, default=None)

args = apply_gin_config(parser, configs=['experiments/configs/bases/benchmark.gin'])

benchmark(prefix=args.prefix, results_only=args.results_only)
