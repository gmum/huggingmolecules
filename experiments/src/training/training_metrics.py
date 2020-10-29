import logging
from typing import Optional, Any

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.classification import auroc


class BatchWeightedLoss(Metric):
    def __init__(
            self,
            compute_on_step: bool = False,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.add_state("all_loss", default=[])
        self.add_state("all_sizes", default=[])

    def update(self, loss: torch.Tensor, batch_size: int) -> None:
        self.all_loss.append(loss)
        self.all_sizes.append(torch.tensor(batch_size, device=loss.device))

    def compute(self):
        sizes = torch.stack(self.all_sizes)
        weights = sizes / torch.sum(sizes)
        losses = torch.stack(self.all_loss)
        return torch.sum(torch.mul(weights, losses))


class AUROC(Metric):
    def __init__(
            self,
            compute_on_step: bool = False,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.add_state("all_preds", default=[])
        self.add_state("all_target", default=[])

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.all_preds.append(preds)
        self.all_target.append(target)

    def compute(self):
        preds = torch.cat(self.all_preds).view(-1)
        target = torch.cat(self.all_target).view(-1)
        try:
            return auroc(preds, target)
        except ValueError:
            logging.warning('AUROC requires both negative and positive samples. Returning None')


class RMSE(Metric):
    def __init__(
            self,
            compute_on_step: bool = False,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.add_state("all_preds", default=[])
        self.add_state("all_target", default=[])

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.all_preds.append(preds)
        self.all_target.append(target)

    def compute(self):
        preds = torch.cat(self.all_preds).view(-1)
        target = torch.cat(self.all_target).view(-1)
        return torch.sqrt(F.mse_loss(preds, target))
