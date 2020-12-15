import argparse

from experiments.src.gin import apply_gin_config
from experiments.src.benchmarking import benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')
parser.add_argument('--prefix', type=str, default=None)

args = apply_gin_config(parser, configs=['experiments/configs/bases/benchmark.gin'])

benchmark(prefix=args.prefix, results_only=args.results_only)
