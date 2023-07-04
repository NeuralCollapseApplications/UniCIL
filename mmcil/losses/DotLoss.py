import torch
import torch.nn as nn

from mmcls.models.builder import LOSSES


@LOSSES.register_module()
class DotLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
            self,
            feat,
            target,
            avg_factor=None,
    ):
        assert avg_factor is None
        assert self.reduction == 'mean'
        num_samples = len(feat)
        assert torch.allclose(torch.norm(feat, p=2, dim=1, keepdim=False),
                              torch.ones((num_samples,), device=feat.device, dtype=feat.dtype))
        assert torch.allclose(torch.norm(target, p=2, dim=1, keepdim=False),
                              torch.ones((num_samples,), device=target.device, dtype=target.dtype))
        dot = torch.sum(feat * target, dim=1)

        loss = (1 - dot).mean()

        return loss * self.loss_weight
