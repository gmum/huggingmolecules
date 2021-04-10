from dataclasses import dataclass
from typing import Tuple, List, Optional, Any

import numpy as np
import torch

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from src.huggingmolecules.featurization.featurization_common_utils import stack_y_list
from src.huggingmolecules.models.models_api import PretrainedModelBase

try:
    import chemprop
except ImportError:
    raise ImportError('Please install chemprop v.1.1.0 (pip install chemprop==1.1.0) '
                      'from https://github.com/chemprop/chemprop to use ChempropModelWrapper.')


@dataclass
class ChempropConfig(PretrainedConfigMixin):
    d_model: int = 300
    features_generators: List[str] = None
    depth: int = 3


@dataclass
class ChempropBatchEncoding(RecursiveToDeviceMixin):
    batch_mol_graph: Any
    batch_features: Optional[List[torch.Tensor]]
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class ChempropFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], ChempropBatchEncoding, ChempropConfig]):
    def __init__(self, config: ChempropConfig):
        super().__init__(config)
        self.features_generators = config.features_generators

    def _collate_encodings(self, encodings: List[Tuple[Any, Optional[np.array], float]]) -> ChempropBatchEncoding:
        mol_graph_list, features_list, y_list = zip(*encodings)
        batch_mol_graph = chemprop.features.BatchMolGraph(mol_graph_list)
        if features_list is not None and all(f is not None for f in features_list):
            batch_features = [torch.tensor(f).float() for f in features_list]
        else:
            batch_features = None
        return ChempropBatchEncoding(batch_mol_graph=batch_mol_graph,
                                     batch_features=batch_features,
                                     y=stack_y_list(y_list),
                                     batch_size=len(y_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[Any, np.array, float]:
        datapoint = chemprop.data.MoleculeDatapoint([smiles], features_generator=self.features_generators)
        mol_graph = chemprop.features.MolGraph(datapoint.mol[0])
        features = datapoint.features
        return mol_graph, features, y


class ChempropModelWrapper(PretrainedModelBase):
    def __init__(self, config: ChempropConfig):
        super().__init__(config)
        args = chemprop.args.TrainArgs()
        args.parse_args(args=["--data_path", "non_existent", "--dataset_type", 'regression'])
        args.task_names = ["whatever"]
        args.depth = config.depth
        args.hidden_size = config.d_model
        self.model = chemprop.models.MoleculeModel(args)

    @classmethod
    def get_featurizer_cls(cls):
        return ChempropFeaturizer

    @classmethod
    def get_config_cls(cls):
        return ChempropConfig

    def forward(self, batch: ChempropBatchEncoding):
        return self.model([batch.batch_mol_graph], batch.batch_features)

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)
