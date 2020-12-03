from experiments.training.training_benchmark import benchmark
from src.huggingmolecules import MatModel, MatFeaturizer
from src.huggingmolecules.utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')
args = apply_gin_config(parser)

model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()

benchmark(model, featurizer, 'MAT', results_only=args.results_only)
