import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from experiments.models.chembert.wrappers import ChemBertFeaturizer, ChemBertModelWrapper
from experiments.training import tune_hyper
from src.huggingmolecules.utils import apply_gin_config

os.environ["TOKENIZERS_PARALLELISM"] = "true"
apply_gin_config()
model = AutoModelForSequenceClassification.from_pretrained("seyonec/ChemBERTa-zinc-base-v1", num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
featurizer = ChemBertFeaturizer(tokenizer)
model = ChemBertModelWrapper(model)

tune_hyper(model, featurizer)
