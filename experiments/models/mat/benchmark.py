from experiments.training.training_benchmark import benchmark
from src.huggingmolecules import MatModel, MatFeaturizer
from src.huggingmolecules.utils import *

apply_gin_config()
model = MatModel.from_pretrained('mat-base-freesolv')
featurizer = MatFeaturizer()

benchmark(model, featurizer, 'MAT')
