from experiments.src.ensembles.ensembles_api import EnsembleElement, EnsembleModule
from experiments.src.evaluation import evaluate_model
from experiments.src.wrappers.wrappers_chemprop import ChempropFeaturizer
from src.huggingmolecules import MatModel
from experiments.src.gin import apply_gin_config

apply_gin_config(base='evaluate')

models = [EnsembleElement(MatModel, 'mat-base-freesolv', 'saved/mat/weights0.ckpt'),
          EnsembleElement(MatModel, 'mat-base-freesolv', 'saved/mat/weights1.ckpt')]
ensemble = EnsembleModule(models)
featurizer = ChempropFeaturizer()

evaluate_model(ensemble, featurizer)
