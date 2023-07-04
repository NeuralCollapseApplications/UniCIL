import math
from typing import Dict

import numpy as np
import torch
from mmcls.utils import get_root_logger

from mmcls.models.heads import ClsHead

from mmcls.models.builder import HEADS


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec


@HEADS.register_module()
class ETFHeadCE(ClsHead):
    """Classification head for Baseline.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes: int, in_channels: int, *args, **kwargs) -> None:

        self.eval_classes = (0, num_classes)

        super().__init__(*args, **kwargs)
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.logger = get_root_logger()

        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.register_buffer('etf_vec', etf_vec)

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def get_eval_classes(self):
        return self.eval_classes

    def forward_train(self, x: torch.Tensor, gt_label: torch.Tensor, **kwargs) -> Dict:
        """Forward training data."""
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes[1]]
        assert self.eval_classes[0] == 0
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def loss(self, cls_score, gt_label, **kwargs):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            cls_score, gt_label, avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss_cls'] = loss
        return losses

    def simple_test(self, x, softmax=False, post_process=False, return_feat=False):
        x = self.pre_logits(x)
        if return_feat:
            return x
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes[1]]
        cls_score[:, :self.eval_classes[0]] = -10.
        assert not softmax
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score

    def get_cls_score(self, x):
        cls_score = x @ self.etf_vec
        return cls_score[:, self.eval_classes[0]:self.eval_classes[1]]
