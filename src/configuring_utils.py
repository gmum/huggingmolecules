import json


class PretrainedConfigMixin:
    @classmethod
    def _get_arch_from_pretrained_name(cls, pretrained_name):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_name):
        file_path = cls._get_arch_from_pretrained_name(pretrained_name)
        with open(file_path, 'r') as fp:
            param_dict = json.load(fp)
        return cls(**param_dict)

    def save(self, file_path):
        with open(file_path, 'w') as fp:
            json.dump(self.__dict__, fp)

