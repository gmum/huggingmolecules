import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from experiments.models.chembert.wrappers import ChemBertFeaturizer, ChemBertModelWrapper
from experiments.training.training_benchmark import benchmark
from src.huggingmolecules.utils import apply_gin_config

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_only', action='store_true')
args = apply_gin_config(parser)

model = AutoModelForSequenceClassification.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
featurizer = ChemBertFeaturizer(tokenizer)
model = ChemBertModelWrapper(model)

benchmark(model, featurizer, 'ChemBert', results_only=args.results_only)
