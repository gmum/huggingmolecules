import unittest

from src.huggingmolecules import MatFeaturizer, MatModel, GroverFeaturizer, GroverModel
from pytorch_lightning import Trainer


class TrainerTest(unittest.TestCase):
    def test_mat_trainer(self):
        model = MatModel.from_pretrained('mat-base-freesolv')
        featurizer = MatFeaturizer()
        dataset = featurizer.load_dataset_from_csv('/home/panjan/Desktop/GMUM/huggingmolecules/data/freesolv/freesolv.csv')
        dataloader = featurizer.get_data_loader(dataset[:5], batch_size=3)

        trainer = Trainer(max_epochs=5)
        trainer.fit(model, train_dataloader=dataloader)

    def test_grover_trainer(self):
        model = GroverModel.from_pretrained('grover-base-whatever')
        featurizer = GroverFeaturizer()
        dataset = featurizer.load_dataset_from_csv('/home/panjan/Desktop/GMUM/huggingmolecules/data/freesolv/freesolv.csv')
        dataloader = featurizer.get_data_loader(dataset[:5], batch_size=3)

        trainer = Trainer(max_epochs=5)
        trainer.fit(model, train_dataloader=dataloader)
