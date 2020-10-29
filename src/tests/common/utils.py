import random

import numpy as np
import torch


def assert_arrays_almost_equals(a, b):
    if a is None or b is None:
        assert a == b
    else:
        if torch.is_tensor(a) or torch.is_tensor(b):
            torch.testing.assert_allclose(a, b)
        elif isinstance(a, list) or isinstance(b, list):
            assert a == b
        else:
            assert np.allclose(a, b)


def assert_encoding_almost_equals(a, b, excluded=None):
    assert_dicts_almost_equals(a.__dict__, b.__dict__, excluded)


def assert_dicts_almost_equals(a, b, excluded=None):
    excluded = excluded if excluded else []
    assert a.keys() == b.keys()
    for key in a.keys():
        if key not in excluded:
            assert_arrays_almost_equals(a[key], b[key])


def assert_negate(assertion):
    try:
        assertion()
    except AssertionError:
        pass


def get_excluded_params(model, head_layers):
    return [p for p in model.state_dict().keys()
            if any(p.split('.')[0] == e for e in head_layers)]


def get_random_config_param(config_cls):
    params = list(config_cls().to_dict().keys())
    return {random.choice(params): random.random()}
