import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.chemformers.callbacks import get_default_callbacks
from sklearn.model_selection import train_test_split

from src.chemformers import MatModel, MatFeaturizer
from src.chemformers import GroverModel, GroverFeaturizer
import numpy as np
import gin
from src.chemformers.utils import *


@gin.configurable(blacklist=['model', 'featurizer'])
def train(model, featurizer, *, save_path, data_path, num_epochs=10, batch_size, gpus, resume, seed=None):
    resume_path = os.path.join(save_path, 'last.ckpt')
    if not resume and os.path.exists(resume_path):
        raise IOError("Please clear folder before running or pass train.resume=True")

    dataset = featurizer.load_dataset_from_csv(data_path)[:10]
    random_state = None if seed is None else np.random.RandomState(seed)
    train_data, val_data = train_test_split(dataset, train_size=0.9, random_state=random_state)

    train_loader, val_loader = featurizer.get_data_loaders(train_data, val_data, batch_size=batch_size, num_workers=4)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(save_path, "weights"),
        verbose=True,
        save_last=True,  # For resumability
        monitor='val_loss',
        mode='min'
    )
    trainer = Trainer(
        default_root_dir=save_path,
        max_epochs=num_epochs,
        callbacks=get_default_callbacks() + [checkpoint_callback],
        log_every_n_steps=1,
        checkpoint_callback=True,
        resume_from_checkpoint=resume_path if resume else None,
        gpus=gpus)

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    apply_gin_config()
    model = MatModel.from_pretrained('mat-base-freesolv')
    featurizer = MatFeaturizer()
    # model = GroverModel.from_pretrained('grover-base-whatever')
    # featurizer = GroverFeaturizer()
    train(model, featurizer)
