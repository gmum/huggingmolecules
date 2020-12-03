import os
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from experiments.models.chembert.wrappers import ChemBertFeaturizer, ChemBertModelWrapper
from experiments.training.training_benchmark import benchmark
from experiments.training.training_benchmark_utils import set_default_experiment_name, print_benchmark_results
from src.huggingmolecules.utils import apply_gin_config

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')
args = apply_gin_config(parser)

name = "ChemBert_2"
if args.results_only:
    set_default_experiment_name(prefix=name)
    print_benchmark_results()
    sys.exit()

model = AutoModelForSequenceClassification.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k")
featurizer = ChemBertFeaturizer(tokenizer)
model = ChemBertModelWrapper(model)
benchmark(model, featurizer, prefix=name)
