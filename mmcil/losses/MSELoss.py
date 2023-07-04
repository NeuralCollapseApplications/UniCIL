import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import LOSSES


@LOSSES.register_module()
class MSELoss(nn.Module):

    def __init__(
            self,
            reduction='mean',
            loss_weight=1.0
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                x1,
                x2,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_mse = self.loss_weight * F.mse_loss(x1, x2, reduction=reduction)
        return loss_mse
