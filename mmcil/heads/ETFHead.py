import math
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist

from mmcv.runner import get_dist_info
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
class ETFHead(ClsHead):
    """Classification head for Baseline.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes: int, in_channels: int, *args, **kwargs) -> None:

        self.eval_classes = (0, num_classes)

        # training settings about different length for different classes
        if kwargs.pop('with_len', False):
            self.with_len = True
        else:
            self.with_len = False

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
        if self.with_len:
            etf_vec = self.etf_vec
            target = (etf_vec * self.produce_training_rect(gt_label, self.num_classes))[:, gt_label].t()
        else:
            target = self.etf_vec[:, gt_label].t()
        losses = self.loss(x, target)
        if self.cal_acc:
            with torch.no_grad():
                cls_score = x @ self.etf_vec
                cls_score = cls_score[:, :self.eval_classes[1]]
                cls_score[:, :self.eval_classes[0]] = -10.
                acc = self.compute_accuracy(cls_score, gt_label)
                assert len(acc) == len(self.topk)
                losses['accuracy'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        return losses

    def loss(self, feat, target, **kwargs):
        losses = dict()
        # compute loss
        if self.with_len:
            loss = self.compute_loss(feat, target, m_norm2=torch.norm(target, p=2, dim=1))
        else:
            loss = self.compute_loss(feat, target)
        losses['loss_dr'] = loss
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

    @staticmethod
    def produce_training_rect(label: torch.Tensor, num_classes: int):
        rank, world_size = get_dist_info()
        if world_size > 1:
            recv_list = [None for _ in range(world_size)]
            dist.all_gather_object(recv_list, label.cpu())
            new_label = torch.cat(recv_list).to(device=label.device)
            label = new_label
        uni_label, count = torch.unique(label, return_counts=True)
        batch_size = label.size(0)
        uni_label_num = uni_label.size(0)
        assert batch_size == torch.sum(count)
        gamma = torch.tensor(batch_size / uni_label_num, device=label.device, dtype=torch.float32)
        rect = torch.ones(1, num_classes).to(device=label.device, dtype=torch.float32)
        rect[0, uni_label] = torch.sqrt(gamma / count)
        return rect
