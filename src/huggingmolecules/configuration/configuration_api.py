import json
import os

from ..downloading.downloading_utils import from_cache


class PretrainedConfigMixin:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def _get_archive_dict(cls) -> dict:
        raise NotImplementedError

    @classmethod
    def _load_dict_from_pretrained(cls, pretrained_name: str):
        archive_dict = cls._get_archive_dict()
        file_path = from_cache(pretrained_name, archive_dict, 'json')
        if not file_path:
            file_path = os.path.expanduser(pretrained_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)

        with open(file_path, 'r') as fp:
            param_dict: dict = json.load(fp)

        return param_dict

    @classmethod
    def from_pretrained(cls, pretrained_name: str, **kwargs):
        param_dict = cls._load_dict_from_pretrained(pretrained_name)

        for k, v in kwargs.items():
            param_dict[k] = v

        return cls(**param_dict)

    def to_dict(self):
        return self.__dict__

    def save(self, file_path: str):
        with open(file_path, 'w') as fp:
            json.dump(self.to_dict(), fp, indent=2)
