import copy
import math
from typing import Dict, Optional

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
class ETFHeadInc(ClsHead):
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

        if kwargs.pop('with_ftc', True):
            self.with_ftc = True
        else:
            self.with_ftc = False

        if kwargs.pop('with_ncm', False):
            self.with_ncm = True
        else:
            self.with_ncm = False

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

        self.register_buffer('etf_proto', copy.deepcopy(etf_vec))
        self.use_proto: Optional[Dict[int, int]] = None
        self.epoch: float = 0.

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def get_eval_classes(self):
        return self.eval_classes

    def forward_train(self, x: torch.Tensor, gt_label: torch.Tensor, **kwargs) -> Dict:
        """Forward training data."""
        x = self.pre_logits(x)
        if self.use_proto is not None:
            proto_vec = self.etf_proto[:, self.use_proto[0]:self.use_proto[1]]
            if self.with_ncm:
                assert not self.with_ftc
                assert torch.allclose(proto_vec, self.etf_vec[:, self.use_proto[0]:self.use_proto[1]])
                proto_vec = self.etf_vec[:, self.use_proto[0]:self.use_proto[1]]
            elif self.with_ftc:
                proto_vec = proto_vec * (1 - self.epoch) + self.etf_vec[:, self.use_proto[0]:self.use_proto[1]] * self.epoch
            else:
                proto_vec = self.etf_vec[:, self.use_proto[0]:self.use_proto[1]]
            proto_vec = proto_vec / torch.norm(proto_vec, p=2, dim=0, keepdim=True)
            etf_vec = torch.cat([self.etf_vec[:, :self.use_proto[0]], proto_vec], dim=1)
        else:
            etf_vec = self.etf_vec[:, :self.eval_classes[1]]
        if self.with_len:
            target = (etf_vec * self.produce_training_rect(gt_label, self.eval_classes[1]))[:, gt_label].t()
        else:
            target = etf_vec[:, gt_label].t()
        losses = self.loss(x, target)
        if self.cal_acc:
            with torch.no_grad():
                cls_score = x @ etf_vec
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
        if self.use_proto is not None:
            proto_vec = self.etf_proto[:, self.use_proto[0]:self.use_proto[1]]
            proto_vec = proto_vec * (1 - self.epoch) + self.etf_vec[:, self.use_proto[0]:self.use_proto[1]] * self.epoch
            proto_vec = proto_vec / torch.norm(proto_vec, p=2, dim=0, keepdim=True)
            etf_vec = torch.cat([self.etf_vec[:, :self.use_proto[0]], proto_vec], dim=1)
        else:
            etf_vec = self.etf_vec
        cls_score = x @ etf_vec
        cls_score = cls_score[:, :self.eval_classes[1]]
        cls_score[:, :self.eval_classes[0]] = -10.
        assert not softmax
        if post_process:
            return self.post_process(cls_score)
        else:
            return cls_score

    def set_proto(self, proto, start, end):
        self.logger.info("[{}] : setting proto from {} to {}.".format(self.__class__.__name__, start, end))
        assert len(proto) == end - start
        for idx, item in enumerate(proto):
            self.etf_proto[:, start + idx] = item
            if self.with_ncm:
                self.etf_vec[:, start + idx] = item
        self.use_proto = (start, end)

    def set_epoch(self, epoch: float):
        self.logger.info("[{}] : epoch : {:.3f}.".format(self.__class__.__name__, epoch))
        self.epoch = epoch

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
