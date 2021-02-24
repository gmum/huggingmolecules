import json
import os


class PretrainedConfigMixin:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def _load_dict_from_pretrained(cls, pretrained_name: str):
        if os.path.exists(pretrained_name):
            file_path = pretrained_name
        else:
            file_path = cls._get_arch_from_pretrained_name(pretrained_name)

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
