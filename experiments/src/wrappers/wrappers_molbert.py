import os
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union

import gin
import numpy as np
import torch
from molbert.apps.finetune import FinetuneSmilesMolbertApp
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase


@dataclass
class MolbertBatchEncoding(RecursiveToDeviceMixin):
    data: dict
    y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


@gin.configurable()
class MolbertFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], MolbertBatchEncoding]):
    def __init__(self, max_size=512):
        self.tokenizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(max_size)

    def _collate_encodings(self, encodings: List[Tuple[np.ndarray, float]]) -> MolbertBatchEncoding:
        smiles_list, y_list = zip(*encodings)
        features_list, valid = self.tokenizer.transform(smiles_list)
        valid_features_list = features_list[valid]

        features = torch.tensor(valid_features_list)
        attention_mask = torch.tensor(np.zeros_like(features))
        attention_mask[features != 0] = 1
        type_ids = torch.zeros_like(features)
        data = {'input_ids': features, 'attention_mask': attention_mask, 'token_type_ids': type_ids}
        y = torch.tensor(np.vstack(y_list)).float()[valid]

        return MolbertBatchEncoding(data, y, len(valid_features_list))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[Union[np.ndarray, str], float]:
        return smiles, y

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        return cls()


@dataclass
class MolbertConfig(PretrainedConfigMixin):
    d_model: int = 1024


class MolbertModelWrapper(PretrainedModelBase):
    def __init__(self, model):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        super().__init__(MolbertConfig())
        self.model = model

    @classmethod
    def get_featurizer_cls(cls):
        return MolbertFeaturizer

    @classmethod
    def get_config_cls(cls):
        return MolbertConfig

    def forward(self, batch):
        return self.model(batch.data)['finetune']

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_name: str, task: str, **kwargs):
        print(pretrained_name)
        raw_args_str = (
            f"--max_seq_length 512 "
            f"--batch_size 0 "
            f"--max_epochs 0 "
            f"--num_workers 0 "
            f"--fast_dev_run 0 "
            f"--train_file none "
            f"--valid_file none "
            f"--test_file none "
            f"--mode regression "
            f"--output_size 1 "
            f"--pretrained_model_path {pretrained_name} "
            f"--label_column none "
            f"--freeze_level 0 "
            f"--gpus 1 "
            f"--learning_rate 0 "
            f"--learning_rate_scheduler none "
            f"--default_root_dir none"
        )

        raw_args = raw_args_str.split(" ")
        args = FinetuneSmilesMolbertApp().parse_args(args=raw_args)
        model = FinetuneSmilesMolbertApp().get_model(args)
        return cls(model)
