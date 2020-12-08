import argparse
import sys

from experiments.training.training_benchmark import benchmark
from experiments.training.training_benchmark_utils import set_default_experiment_name, print_benchmark_results
from experiments.training.training_train_model_utils import get_model_and_featurizer
from src.huggingmolecules.utils import apply_gin_config

parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')
parser.add_argument('--prefix', type=str, required=True)
args = apply_gin_config(parser, configs=['experiments/configs/bases/benchmark.gin'])

if args.results_only:
    set_default_experiment_name(prefix=args.prefix)
    print_benchmark_results()
    sys.exit()

model, featurizer = get_model_and_featurizer()

benchmark(model, featurizer, prefix=args.prefix)
