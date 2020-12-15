from typing import Optional, Any, Sequence
import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.classification import auroc


class AUROC(Metric):
    def __init__(
            self,
            sample_weight: Optional[Sequence] = None,
            compute_on_step: bool = False,  # True likely crashes if not every batch contains all classes
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.sample_weight = sample_weight
        self.add_state("all_preds", default=[])
        self.add_state("all_target", default=[])

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.all_preds.append(preds)
        self.all_target.append(target)

    def compute(self):
        preds_tensor = torch.cat(self.all_preds)
        target_tensor = torch.cat(self.all_target)
        return auroc(preds_tensor.view(-1), target_tensor.view(-1), sample_weight=self.sample_weight)
