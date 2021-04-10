from itertools import chain, combinations
from typing import List, Optional, Iterable, TypeVar, Iterator, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch

from experiments.src.benchmarking.benchmarking_utils import EnsembleElement
from experiments.src.training.training_utils import get_metric_cls, get_loss_fn

_T = TypeVar('_T')


def pick_best_ensemble(models_list: List[EnsembleElement], *,
                       targets: dict,
                       max_size: int,
                       names_list: List[str],
                       method: str) -> List[EnsembleElement]:
    if method == 'brute':
        return _pick_best_ensemble_brute(models_list, targets=targets, max_size=max_size)
    elif method == 'greedy':
        return _pick_best_ensemble_greedy(models_list, targets=targets, max_size=max_size)
    elif method == 'all':
        if not max_size:
            return models_list
        size = max_size // len(names_list)
        split = {k: [model for model in models_list if model.name == k][:size] for k in names_list}
        return [model for sublist in split.values() for model in sublist]
    else:
        raise NotImplementedError


def _print_benchmark_1_results(ensemble: List[EnsembleElement],
                               targets: dict):
    print(f'Best ensemble:')
    for model in ensemble:
        print(f'  {model}')

    loss_fn = get_loss_fn()
    metric_cls = get_metric_cls()
    metric_name = metric_cls.__name__.lower()

    results = []
    for name, metric, phase in [('valid_loss', loss_fn, 'valid'),
                                ('test_loss', loss_fn, 'test'),
                                (f'valid_{metric_name}', metric_cls(), 'valid'),
                                (f'test_{metric_name}', metric_cls(), 'test')]:
        mean, std = _evaluate_ensemble(ensemble, targets=targets, phase=phase, metric_fn=metric)
        results.append((name, mean, std))

    results = pd.DataFrame(results, columns=['name', 'mean', 'std'])
    print(results.groupby(['name']).agg({'mean': 'max', 'std': 'max'}))
    print()
    print(f'Result: {round(mean, 3):.3f} \u00B1 {round(std, 3):.3f}')


# picking best ensemble

def _pick_best_ensemble_greedy(models_list: List[EnsembleElement], *,
                               targets,
                               max_size: Optional[int] = None) -> List[EnsembleElement]:
    n = max_size if max_size else len(models_list)

    metric_fn = get_metric_cls()()
    agg_fn = min if metric_fn.direction == 'minimize' else max

    ensemble = []
    losses = []
    best_loss = float('inf')
    for i in range(n):
        for idx, model in enumerate(models_list):
            ensemble.append(model)
            valid_loss, std = _evaluate_ensemble(ensemble, targets=targets, phase='valid', metric_fn=metric_fn)
            losses.append((valid_loss, idx))
            ensemble.pop()
        curr_best_loss, curr_best_idx = agg_fn(losses)
        if agg_fn(best_loss, curr_best_loss) and curr_best_loss != best_loss:
            ensemble.append(models_list[curr_best_idx])
            models_list.pop(curr_best_idx)
            best_loss = curr_best_loss
        else:
            break

    return ensemble


def powerset(iterable: Iterable[_T]) -> Iterator[Tuple[_T, ...]]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def _pick_best_ensemble_brute(models_list: List[EnsembleElement], *,
                              targets: dict,
                              max_size: int = None) -> List[EnsembleElement]:
    metric_fn = get_metric_cls()()
    agg_fn = min if metric_fn.direction == 'minimize' else max

    max_size = max_size if max_size else len(models_list)
    ensembles_list = [list(e) for e in powerset(models_list) if 0 < len(e) <= max_size]

    results = []
    for ensemble in ensembles_list:
        valid_loss, _ = _evaluate_ensemble(ensemble, targets=targets, phase='valid', metric_fn=metric_fn)
        results.append((valid_loss, ensemble))

    best_valid_loss, best_ensemble = agg_fn(results)
    return best_ensemble


def _evaluate_ensemble(ensemble: List[EnsembleElement], *,
                       targets: dict,
                       phase: str,
                       metric_fn: callable) -> Tuple[float, float]:
    results = []
    for seed in targets[phase].keys():
        outputs = [e.outputs[phase][seed] for e in ensemble]
        if isinstance(metric_fn, pl.metrics.Metric):
            outputs = [torch.mean(torch.stack(output), dim=0)
                       if isinstance(output, (tuple, list))
                       else output for output in outputs]

        mean_output = outputs[0] if len(outputs) < 2 else torch.mean(torch.stack(outputs), dim=0)
        res = metric_fn(mean_output, targets[phase][seed])
        results.append(res)

    results = torch.stack(results)
    mean = float(torch.mean(results))
    std = float(torch.std(results))
    return mean, std
