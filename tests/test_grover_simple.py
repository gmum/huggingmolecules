import unittest

from src.chemformers import GroverFeaturizer, GroverModel


class GroverModelTest(unittest.TestCase):

    def test_from_pretrained(self):
        featurizer = GroverFeaturizer()
        batch = featurizer.__call__(["CO", "CSC"])

        model = GroverModel.from_pretrained('grover-base-whatever')
        output = model(batch)
        print(output)
