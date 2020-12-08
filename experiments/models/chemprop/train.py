from experiments.models.chemprop.wrappers import ChemPropFeaturizer, ChemPropModelWrapper
from experiments.training import train_model
from src.huggingmolecules.utils import apply_gin_config

import chemprop

apply_gin_config()
args = chemprop.args.TrainArgs()
args.parse_args(args=["--data_path", "non_existent", "--dataset_type", "regression"])
args.task_names = ["whatever"]
model = chemprop.models.MoleculeModel(args)
model = ChemPropModelWrapper(model)
featurizer = ChemPropFeaturizer()

train_model(model, featurizer)
