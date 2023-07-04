# Copyright (c) OpenMMLab. All rights reserved.
# Copy and modified from mmcls@71ef7ba
import copy
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, Fp16OptimizerHook,
                         build_optimizer, build_runner, get_dist_info)

from mmcil.hooks.save_feat_hook import SaveFeatHook
from mmcil.hooks.set_epoch_hook import SetEpochHook
from mmcls.core import DistOptimizerHook
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.utils import (get_root_logger, wrap_distributed_model,
                         wrap_non_distributed_model)

from mmcil.apis.eval_hook import EvalHook, DistEvalHook
from mmcil.apis.test_fn import single_gpu_test, multi_gpu_test

from mmcv.runner.priority import get_priority


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                val_dataset_cfg=None,
                distributed=False,
                validate=False,
                timestamp=None,
                device=None,
                meta=None):
    """Train a model.

    This method will build dataloaders, wrap the model and build a runner
    according to the provided config.

    Args:
        val_dataset_cfg: validation dataset configures.
        model (:obj:`torch.nn.Module`): The model to be run.
        dataset (:obj:`mmcls.datasets.BaseDataset` | List[BaseDataset]):
            The dataset used to train the model. It can be a single dataset,
            or a list of dataset with the same length as workflow.
        cfg (:obj:`mmcv.utils.Config`): The configs of the experiment.
        distributed (bool): Whether to train the model in a distributed
            environment. Defaults to False.
        validate (bool): Whether to do validation with
            :obj:`mmcv.runner.EvalHook`. Defaults to False.
        timestamp (str, optional): The timestamp string to auto generate the
            name of log files. Defaults to None.
        device (str, optional): TODO
        meta (dict, optional): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
    """
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=cfg.ipu_replicas if device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
        seed=cfg.get('seed'),
        sampler_cfg=cfg.get('sampler', None),
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = wrap_distributed_model(
            model,
            cfg.device,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = wrap_non_distributed_model(
            model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if device == 'ipu':
        if not cfg.runner['type'].startswith('IPU'):
            cfg.runner['type'] = 'IPU' + cfg.runner['type']
        if 'options_cfg' not in cfg.runner:
            cfg.runner['options_cfg'] = {}
        cfg.runner['options_cfg']['replicationFactor'] = cfg.ipu_replicas
        cfg.runner['fp16_cfg'] = cfg.get('fp16', None)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        if device == 'ipu':
            from mmcv.device.ipu import IPUFp16OptimizerHook
            optimizer_config = IPUFp16OptimizerHook(
                **cfg.optimizer_config,
                loss_scale=fp16_cfg['loss_scale'],
                distributed=distributed)
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config,
                loss_scale=fp16_cfg['loss_scale'],
                distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed and (cfg.runner['type'] == 'EpochBasedRunner' or cfg.runner['type'] == 'FastEpochBasedRunner'):
        runner.register_hook(DistSamplerSeedHook())
    if cfg.incremental_settings.vec_init:
        # should be set before eval hook
        # to enable network test without training.
        runner.register_hook(SetEpochHook(), priority=get_priority('LOW') + 1)
    if cfg.save_state:
        runner.register_hook(SaveFeatHook(cfg=cfg, interval=10), priority=get_priority('LOW') + 3)

    # register eval hooks
    if validate:
        for idx, _val_dataset_cfg in enumerate(val_dataset_cfg):
            # val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataset = build_dataset(copy.deepcopy(_val_dataset_cfg), dict(test_mode=True))
            # The specific dataloader settings
            val_loader_cfg = {
                **loader_cfg,
                'shuffle': False,  # Not shuffle by default
                'sampler_cfg': None,  # Not use sampler by default
                **cfg.data.get('val_dataloader', {}),
            }
            val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
            eval_cfg = cfg.get('evaluation', {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            test_fn = multi_gpu_test if distributed else single_gpu_test
            # `EvalHook` needs to be executed after `IterTimerHook`.
            # Otherwise, it will cause a bug if use `IterBasedRunner`.
            # Refers to https://github.com/open-mmlab/mmcv/issues/1261
            runner.register_hook(
                eval_hook(val_dataloader, test_fn=test_fn, **eval_cfg), priority=get_priority('LOW') + 3)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    if runner.max_epochs > 0:
        runner.run(data_loaders, cfg.workflow)
    else:
        runner.logger.info("Skipped train")
        runner.mode = 'train'
        runner.call_hook('before_run')
        runner.call_hook('after_train_epoch')
        runner.call_hook('after_run')
    # Print the results
    if runner.rank == 0:
        final_acc = 0.
        final_acc_cnt = 0.
        acc_str = "[ACC] "
        for hook in runner._hooks:
            if hasattr(hook, 'final_result_hy'):
                acc = hook.final_result_hy
                cls_start, cls_end = hook.dataloader.dataset.get_eval_classes()
                length = len(hook.dataloader.dataset)
                acc_str += '{:.2f} '.format(acc)
                logger.info("{} to {} ; acc : {:.2f} ; len : {}".format(cls_start, cls_end, acc, length))
                final_acc += acc * length
                final_acc_cnt += length
        logger.info(acc_str)
        logger.info("[ACC_MEAN] : {:.2f}".format(final_acc / final_acc_cnt))
