from experiments.ensembles.ensembles_api import EnsembleElement, EnsembleModule
from experiments.training.training_evaluate_model import evaluate_model
from experiments.wrappers.wrappers_chemprop import ChempropFeaturizer
from src.huggingmolecules import MatModel
from src.huggingmolecules.utils import apply_gin_config

apply_gin_config(configs=['experiments/configs/bases/evaluate.gin'])

models = [EnsembleElement(MatModel, 'mat-base-freesolv', 'saved/mat/weights0.ckpt'),
          EnsembleElement(MatModel, 'mat-base-freesolv', 'saved/mat/weights1.ckpt')]
ensemble = EnsembleModule(models)
featurizer = ChempropFeaturizer()

evaluate_model(ensemble, featurizer)
