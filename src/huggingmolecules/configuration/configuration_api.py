import json


class PretrainedConfigMixin:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name: str, **kwargs):
        file_path = cls._get_arch_from_pretrained_name(pretrained_name)
        with open(file_path, 'r') as fp:
            param_dict: dict = json.load(fp)
        for k, v in kwargs:
            param_dict[k] = v
        return cls(**param_dict)

    def get_dict(self):
        return self.__dict__

    def save(self, file_path: str):
        with open(file_path, 'w') as fp:
            json.dump(self.get_dict(), fp, indent=2)
