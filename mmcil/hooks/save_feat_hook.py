import copy
import os.path as osp

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import get_dist_info, weights_to_cpu
from mmcv.runner.checkpoint import get_state_dict
from mmcv.runner.hooks import HOOKS, Hook

from mmcil.utils.func import extract_features


def save_feat(cfg, runner, session_id, epoch_id):
    rank, world_size = get_dist_info()
    if is_module_wrapper(runner.model):
        _model = copy.deepcopy(runner.model.module)
    else:
        _model = copy.deepcopy(runner.model)
    results = extract_features(cfg, _model, 0, cfg.total_cls_num)
    if rank == 0:
        _model.backbone = None
        save_state = {
            'model': weights_to_cpu(get_state_dict(_model)),
            'features': weights_to_cpu(results)
        }
        torch.save(save_state, osp.join(cfg.work_dir, "s{:02d}_{:03d}.pth".format(session_id, epoch_id)))


@HOOKS.register_module()
class SaveFeatHook(Hook):
    def __init__(self, cfg, interval=20):
        self.cfg = cfg
        self.interval = interval

    def before_train_epoch(self, runner):
        if runner.epoch == 0:
            runner.logger.info("[{}]: Saving Features when epoch is 0.".format(self.__class__.__name__))
            save_feat(self.cfg, runner, session_id=runner.meta['session_id'], epoch_id=0)

    def after_train_epoch(self, runner):
        if (runner.epoch + 1) % self.interval == 0:
            runner.logger.info("[{}]: Saving Features when epoch is {}.".format(
                self.__class__.__name__, runner.epoch + 1))
            save_feat(self.cfg, runner, session_id=runner.meta['session_id'], epoch_id=runner.epoch + 1)
