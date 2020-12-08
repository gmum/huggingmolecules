import argparse
import sys

from experiments.models.chemprop.wrappers import ChemPropFeaturizer, ChemPropModelWrapper
from experiments.training import train_model
from experiments.training.training_benchmark import benchmark
from experiments.training.training_benchmark_utils import set_default_experiment_name, print_benchmark_results
from src.huggingmolecules.utils import apply_gin_config

import chemprop

prefix = "Chemprop_vanilla3"

parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')
args = apply_gin_config(parser)

if args.results_only:
    set_default_experiment_name(prefix=prefix)
    print_benchmark_results()
    sys.exit()

args = chemprop.args.TrainArgs()
args.parse_args(args=["--data_path", "non_existent", "--dataset_type", "regression"])
args.task_names = ["whatever"]
model = chemprop.models.MoleculeModel(args)
model = ChemPropModelWrapper(model)
featurizer = ChemPropFeaturizer()

benchmark(model, featurizer, prefix=prefix)
