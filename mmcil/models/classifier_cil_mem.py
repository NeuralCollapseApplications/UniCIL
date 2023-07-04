# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
# from torch.nn.modules.batchnorm import _BatchNorm

from mmcls.models.builder import CLASSIFIERS, build_backbone, build_head, build_neck, build_loss
from mmcls.models.heads import MultiLabelClsHead
from mmcls.models.utils.augment import Augments
from mmcls.models.classifiers.base import BaseClassifier


@CLASSIFIERS.register_module()
class ImageClassifierCILWithMemory(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 loss_dist=None,
                 loss_dist2=None,
                 dist_mode='pre_logits',
                 train_cfg=None,
                 init_cfg=None,
                 with_pod=False):
        super().__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

        self.pre_backbone = None
        self.pre_neck = None
        self.eval_list = ('pre_backbone', 'pre_neck')
        self.dist_loss = build_loss(loss_dist) if loss_dist is not None else None
        self.dist_loss2 = build_loss(loss_dist2) if loss_dist2 is not None else None
        self.dist_mode = dist_mode
        self.with_pod = with_pod

    def extract_feat(self, img, stage='neck'):
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(img)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        if self.with_head and hasattr(self.head, 'pre_logits'):
            x = self.head.pre_logits(x)
        return x

    def extract_feat_pre(self, img, stage='neck'):
        assert stage in ['neck', 'backbone']
        x = self.pre_backbone(img)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.pre_neck(x)
        if stage == 'neck':
            return x
        return x

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, lam, gt_label, gt_label_aux = self.augments(img, gt_label)
        else:
            gt_label_aux = None
            lam = None

        losses = dict()
        if self.with_pod:
            inputs = self.extract_feat(img, stage='backbone')
            x_pre = None
            if self.dist_loss is not None and self.pre_backbone is not None:
                inputs_pre = self.extract_feat_pre(img, stage='backbone')
                assert len(inputs) == len(inputs_pre)
                loss_func = self.dist_loss2 if self.dist_loss2 is not None else self.dist_loss
                for idx in range(len(inputs)):
                    _x = inputs[idx]
                    _x_pre = inputs_pre[idx]
                    for mean_id in [-2, -1]:
                        x = _x.mean(dim=mean_id).transpose(1, 2).flatten(0, 1)
                        x_pre = _x_pre.mean(dim=mean_id).transpose(1, 2).flatten(0, 1)
                        if self.dist_mode == 'pre_logits':
                            loss = loss_func(self.head.pre_logits(x), self.head.pre_logits(x_pre).detach())
                        elif self.dist_mode == 'pre_logits_old':
                            indices = torch.lt(gt_label, self.head.use_proto[0])
                            if len(x) > len(indices):
                                assert len(x) % len(indices) == 0
                                indices = indices.repeat(len(x) // len(indices))
                            if indices.any():
                                loss = loss_func(self.head.pre_logits(x)[indices],
                                                 self.head.pre_logits(x_pre)[indices].detach())
                            else:
                                loss = torch.tensor(0., device=x.device)
                        elif self.dist_mode == 'get_cls_score':
                            loss = loss_func(self.head.get_cls_score(x), self.head.get_cls_score(x_pre).detach())
                        elif self.dist_mode == 'none':
                            loss = loss_func(x, x_pre.detach())
                        else:
                            raise NotImplementedError
                        if mean_id == -2:
                            losses.update({'loss_dist_s{}_h'.format(idx + 1): loss})
                        elif mean_id == -1:
                            losses.update({'loss_dist_s{}_w'.format(idx + 1): loss})
                        else:
                            raise ValueError
                x_pre = self.neck(inputs_pre) if self.with_neck else inputs_pre
            x = self.neck(inputs) if self.with_neck else inputs
        else:
            x = self.extract_feat(img)
            x_pre = None

        if self.dist_loss is not None and self.pre_backbone is not None:
            if x_pre is None:
                x_pre = self.extract_feat_pre(img)
            if self.dist_mode == 'pre_logits':
                loss = self.dist_loss(self.head.pre_logits(x), self.head.pre_logits(x_pre).detach())
            elif self.dist_mode == 'pre_logits_old':
                indices = torch.lt(gt_label, self.head.use_proto[0])
                if indices.any():
                    loss = self.dist_loss(self.head.pre_logits(x)[indices], self.head.pre_logits(x_pre)[indices].detach())
                else:
                    loss = torch.tensor(0., device=x.device)
            elif self.dist_mode == 'get_cls_score':
                loss = self.dist_loss(self.head.get_cls_score(x), self.head.get_cls_score(x_pre).detach())
            elif self.dist_mode == 'none':
                loss = self.dist_loss(x, x_pre.detach())
            else:
                raise NotImplementedError
            losses.update({'loss_dist': loss})

        loss = self.head.forward_train(x, gt_label)

        losses.update(loss)

        if gt_label_aux is not None:
            aux_loss = self.head.forward_train(x, gt_label_aux)
            for k in aux_loss:
                if k.startswith('loss'):
                    losses[k + '_main'] = losses[k] * lam
                    losses[k + '_aux'] = aux_loss[k] * (1 - lam)
                    del losses[k]

        return losses

    def simple_test(self, img, gt_label=None, img_metas=None, return_acc=False, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img)

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmax' not in kwargs, (
                'Please use `sigmoid` instead of `softmax` '
                'in multi-label tasks.')
        res = self.head.simple_test(x, post_process=not return_acc, **kwargs)
        if return_acc:
            assert gt_label is not None
            res = res.argmax(dim=-1)
            return torch.eq(res, gt_label).to(dtype=torch.float32).cpu().numpy().tolist()
        return res

    def save_pre_model(self):
        self.pre_backbone = copy.deepcopy(self.backbone)
        for param in self.pre_backbone.parameters():
            param.requires_grad = False
        self.pre_backbone.train(False)
        self.pre_neck = copy.deepcopy(self.neck)
        for param in self.pre_neck.parameters():
            param.requires_grad = False
        self.pre_neck.train(False)
        # for n, m in self.named_modules():
        #     if isinstance(m, _BatchNorm):
        #         if not n.split('.')[0] in self.eval_list:
        #             m.reset_running_stats()

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if name not in self.eval_list:
                module.train(mode)
            else:
                module.train(False)
        return self
