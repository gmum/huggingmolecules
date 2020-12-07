import argparse
import sys

from experiments.training.training_benchmark import benchmark
from experiments.training.training_benchmark_utils import set_default_experiment_name, print_benchmark_results
from src.huggingmolecules import MatModel, MatFeaturizer
from src.huggingmolecules.utils import apply_gin_config

name = "MAT_linear"

parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')
args = apply_gin_config(parser)

if args.results_only:
    set_default_experiment_name(prefix=name)
    print_benchmark_results()
    sys.exit()

model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()

benchmark(model, featurizer, name)
