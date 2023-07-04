# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil

import mmcv
import numpy as np
import platform
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):
    """Non-Distributed evaluation hook.
    """

    def __init__(self, dataloader, **kwargs):
        super(EvalHook, self).__init__(dataloader, **kwargs)
        self.final_result_hy = None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        results = self.test_fn(runner.model, self.dataloader)
        if runner.rank == 0:
            runner.logger.info("Eval Hook : There are {} samples in total.".format(len(results)))
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        final_result = np.array(results).mean() * 100
        key_score = {
            "acc_{}_{}".format(*self.dataloader.dataset.get_eval_classes()): final_result
        }
        self.final_result_hy = final_result

        for name, val in key_score.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        if self.save_best:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(key_score.keys())[0])
            self._save_ckpt(runner, key_score[self.key_indicator])

            dst_file = osp.join(self.out_dir, 'best.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(osp.basename(self.best_ckpt_path), dst_file)
            else:
                shutil.copy(osp.basename(self.best_ckpt_path), dst_file)


class DistEvalHook(BaseDistEvalHook):
    """Non-Distributed evaluation hook.

    Comparing with the ``EvalHook`` in MMCV, this hook will save the latest
    evaluation results as an attribute for other hooks to use (like
    `MMClsWandbHook`).
    """

    def __init__(self, dataloader, **kwargs):
        super(DistEvalHook, self).__init__(dataloader, **kwargs)
        self.final_result_hy = None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        results = self.test_fn(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print()
            runner.logger.info("Dist Eval Hook : There are {} samples in total. {} to {}".format(
                len(results), *self.dataloader.dataset.get_eval_classes()))
        if runner.rank == 0:
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            final_result = np.array(results).mean() * 100
            key_score = {
                "acc_{}_{}".format(*self.dataloader.dataset.get_eval_classes()): final_result
            }
            self.final_result_hy = final_result

            for name, val in key_score.items():
                runner.log_buffer.output[name] = val
            runner.log_buffer.ready = True
            if self.save_best:
                if self.key_indicator == 'auto':
                    # infer from eval_results
                    self._init_rule(self.rule, list(key_score.keys())[0])
                self._save_ckpt(runner, key_score[self.key_indicator])

                dst_file = osp.join(self.out_dir, 'best.pth')
                if platform.system() != 'Windows':
                    mmcv.symlink(osp.basename(self.best_ckpt_path), dst_file)
                else:
                    shutil.copy(osp.basename(self.best_ckpt_path), dst_file)

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif self.start == 0 and current == 0 and self.initial_flag:
            # Added by HY, let the network to eval before train when start is 0
            return True
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True
