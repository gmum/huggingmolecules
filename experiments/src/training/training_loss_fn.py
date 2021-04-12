import gin
import torch
import torch.nn as nn


class GroverLoss:
    def __init__(self, dist_coff: float):
        task_name = gin.query_parameter('task.name')
        if task_name == 'classification':
            self.pred_loss = nn.BCEWithLogitsLoss()
        elif task_name == 'regression':
            self.pred_loss = nn.MSELoss()
        else:
            raise ValueError(f'Task type "{task_name}" not supported.')

        self.dist_loss = nn.MSELoss()
        self.dist_coff = dist_coff

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_1, pred_2 = preds
        dist_loss = self.dist_loss(torch.sigmoid(pred_1), torch.sigmoid(pred_2))
        pred_loss_1 = self.pred_loss(pred_1, target)
        pred_loss_2 = self.pred_loss(pred_2, target)
        return pred_loss_1 + pred_loss_2 + self.dist_coff * dist_loss
