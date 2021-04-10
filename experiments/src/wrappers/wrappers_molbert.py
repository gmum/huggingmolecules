import os
from dataclasses import dataclass
from typing import Tuple, List, Optional, Type, Any

import numpy as np
import torch

from src.huggingmolecules.configuration.configuration_api import PretrainedConfigMixin
from src.huggingmolecules.featurization.featurization_api import PretrainedFeaturizerMixin, RecursiveToDeviceMixin
from src.huggingmolecules.models.models_api import PretrainedModelBase

try:
    import molbert
except ImportError:
    raise ImportError('Please install molbert from the fork https://github.com/panpiort8/MolBERT '
                      'to use MolbertModelWrapper.')


@dataclass
class MolbertConfig(PretrainedConfigMixin):
    max_size: int = 512

    @classmethod
    def from_pretrained(cls, pretrained_name: str, **kwargs):
        return cls(**kwargs)


@dataclass
class MolbertBatchEncoding(RecursiveToDeviceMixin):
    data: dict
    y: torch.FloatTensor
    invalid_y: torch.FloatTensor
    batch_size: int

    def __len__(self):
        return self.batch_size


class MolbertFeaturizer(PretrainedFeaturizerMixin[Tuple[dict, float], MolbertBatchEncoding, MolbertConfig]):
    def _unsalt(self, smiles: str) -> str:
        return smiles.replace('[Na+].', '').replace('.[Na+]', '')

    def _unsalt_smiles(self, smiles_list: List[str]):
        unstalted_smiles_list = [self._unsalt(smiles) for smiles in smiles_list]
        return self.tokenizer.transform(unstalted_smiles_list)

    def __init__(self, config: MolbertConfig):
        super().__init__(config)
        from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
        self.tokenizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(config.max_size)

    def _collate_encodings(self, encodings: List[Tuple[Any, Any, float]]) -> MolbertBatchEncoding:
        features_list, valid_list, y_list = zip(*encodings)

        features = np.vstack(features_list)
        valid = np.hstack(valid_list)
        y = np.array(y_list, dtype=float)

        valid_features = features[valid]
        valid_y = y[valid]
        invalid_y = y[np.logical_not(valid)]

        features = torch.tensor(valid_features)
        valid_y = torch.tensor(valid_y).view(-1, 1)
        invalid_y = torch.tensor(invalid_y).float()
        attention_mask = torch.tensor(np.zeros_like(features))
        attention_mask[features != 0] = 1
        type_ids = torch.zeros_like(features)
        data = {'input_ids': features, 'attention_mask': attention_mask, 'token_type_ids': type_ids}

        return MolbertBatchEncoding(data=data,
                                    y=valid_y,
                                    invalid_y=invalid_y,
                                    batch_size=len(valid_features))

    def _encode_smiles(self, smiles: str, y: Optional[float]) -> Tuple[Any, Any, float]:
        indices_array, valid = self.tokenizer.transform_single(smiles)
        return indices_array, valid, y


class MolbertModelWrapper(PretrainedModelBase[MolbertBatchEncoding, MolbertConfig]):
    @classmethod
    def get_featurizer_cls(cls) -> Type[MolbertFeaturizer]:
        return MolbertFeaturizer

    @classmethod
    def get_config_cls(cls) -> Type[MolbertConfig]:
        return MolbertConfig

    def forward(self, batch):
        return self.model(batch.data)['finetune']

    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name: str, *,
                        excluded: List[str] = None,
                        config: MolbertConfig = None) -> "MolbertModelWrapper":
        if not config:
            config = MolbertConfig()

        raw_args_str = (
            f"--max_seq_length {config.max_size} "
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
        from molbert.apps.finetune import FinetuneSmilesMolbertApp
        try:
            args = FinetuneSmilesMolbertApp().parse_args(args=raw_args)
            model = FinetuneSmilesMolbertApp().get_model(args)
        except FileNotFoundError as error:
            print(error)
            raise RuntimeError('Please follow README.md from fork https://github.com/panpiort8/MolBERT '
                               'to download the pretrained MolBERT model')
        return cls(model, config)
