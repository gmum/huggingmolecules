from pytorch_lightning import Trainer

from .training_lightning_module import TrainingModule
from .training_train_model_utils import *


@gin.configurable('evaluate', blacklist=['model', 'featurizer'])
def evaluate_model(model: PretrainedModelBase,
                   featurizer: PretrainedFeaturizerMixin,
                   save_path: str,
                   gpus: List[int]):
    trainer = Trainer(default_root_dir=save_path,
                      gpus=gpus)

    loss_fn = get_loss_fn()
    pl_module = TrainingModule(model, optimizer=None, loss_fn=loss_fn)

    _, _, test_loader = get_data_loaders(featurizer)

    return trainer.test(pl_module, test_loader)
