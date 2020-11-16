import json
from typing import *

T = TypeVar('T')
T_Config = TypeVar("T_Config")


# should be f-bounded
# meh. Typing in python sucks...
# https://github.com/AlexandreDecan/portion/issues/27


class PretrainedConfigMixin:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name: str):
        file_path = cls._get_arch_from_pretrained_name(pretrained_name)
        with open(file_path, 'r') as fp:
            param_dict: dict = json.load(fp)
        return cls(**param_dict)

    def save(self, file_path: str):
        with open(file_path, 'w') as fp:
            json.dump(self.__dict__, fp)
