import torch
import torch.nn as nn

from mmcls.models.builder import LOSSES


@LOSSES.register_module()
class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        else:
            h_norm2 = h_norm2.detach()
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        else:
            m_norm2 = m_norm2.detach()

        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)

        return loss * self.loss_weight
