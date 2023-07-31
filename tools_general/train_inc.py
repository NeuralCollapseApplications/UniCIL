# Copyright (c) OpenMMLab. All rights reserved.
# Copy and modified from mmcls@71ef7ba
import argparse
import copy
import math
import os
import os.path as osp
import time
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.runner.checkpoint import load_checkpoint

from mmcil.utils.dist_utils import dist_check
from mmcls import __version__
from mmcls.apis import init_random_seed, set_random_seed
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, collect_env, get_root_logger,
                         setup_multi_processes)

import mmcil  # noqa: F401
from mmcil.apis.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--device', help='device used for training. (Deprecated)')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--ipu-replicas',
        type=int,
        default=None,
        help='num of ipu replicas to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.ipu_replicas is not None:
        cfg.ipu_replicas = args.ipu_replicas
        args.device = 'ipu'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = args.device or auto_select_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    model = build_classifier(cfg.model)
    model.init_weights()

    if len(cfg.workflow) == 2:
        raise ValueError("Not implemented yet.")
    base_cls = cfg.base_cls_num
    total_cls = cfg.total_cls_num
    step = cfg.cls_num_step
    num_sessions = (total_cls - base_cls) // step

    base_dataset_cfg = copy.deepcopy(cfg.data.train)
    base_dataset_cfg.cls_range = (0, base_cls)
    datasets = build_dataset(base_dataset_cfg)

    val_dataset_cfg = copy.deepcopy(cfg.data.val)
    val_dataset_cfg.cls_range = (0, base_cls)
    val_datasets = [val_dataset_cfg]

    logger.info("Starting base session cls num : {}".format(base_cls))
    model.head.eval_classes = (0, base_cls)
    logger.info("Model is evaluating {} to {}".format(*model.head.get_eval_classes()))
    meta.update(
        dict(
            mmcls_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets.CLASSES)
    )
    meta['session_id'] = 0
    
    result_ckpt_filename = "base_ckpt.pth"
    load_path = osp.join(cfg.work_dir, result_ckpt_filename)
    load_checkpoint(model, load_path, logger=logger, map_location='cpu', strict=True)
    logger.info("Model loaded from {}".format(load_path))
    if cfg.incremental_settings.init_proto:
        from mmcil.heads import ProtoHead
        proto_head = ProtoHead(
            in_channels=model.head.in_channels,
            num_classes=total_cls
        )
        model.head = proto_head
        logger.info("Init proto.")
        cls_means_inc = datasets.extract_prototype(cfg, model_inf=model)
        model.head.set_proto(
            cls_means_inc, start=0, end=base_cls
        )
        dist_check(model.head.etf_proto)

    # lets do the incremental sessions
    logger.info("Getting exemplars.... , ")
    datasets.extract_exemplars(cfg, model)
    logger.info("...Done!, datasets is with {} len now.".format(len(datasets)))
    datasets_memory = [datasets]
    logger.info("Base session done.")
    if hasattr(model, 'save_pre_model'):
        logger.info("Saving previous model....")
        model.save_pre_model()
    logger.info("Update config for incremental sessions.")
    cfg.merge_from_dict(cfg.incremental_update)
    if cfg.incremental_settings.lr is not None:
        model.head.compute_loss.loss_weight = cfg.incremental_settings.lr
        logger.info("Update lr_dr to {}.".format(model.head.compute_loss.loss_weight))
    if cfg.incremental_settings.fix_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.norm_eval = True
        logger.info("Fix backbone.")
    logger.info(f'Config:\n{cfg.pretty_text}')
    SEED_INC = cfg.general_settings.seed + (2**31-1)
    rng1 = np.random.default_rng(SEED_INC)
    session_types = rng1.choice(['plain', 'lt', 'fs'], num_sessions, replace=True)
    logger.info("[Session Types]: {}".format(session_types))
    for session_id in range(num_sessions):
        session_type = session_types[session_id]
        logger.info("Starting session {} with type {}".format(session_id + 1, session_type))
        session_dataset_cfg = copy.deepcopy(cfg.data.train)
        session_dataset_cfg.cls_range = (base_cls + session_id * step, base_cls + (session_id + 1) * step)
        _cfg = copy.deepcopy(cfg)
        if session_type == 'plain':
            pass
        elif session_type == 'lt':
            session_dataset_cfg['is_lt'] = True
            session_dataset_cfg['lt_factor'] = cfg.general_settings.lt_factor
            session_dataset_cfg['lt_shuffle'] = True
        elif session_type == 'fs':
            session_dataset_cfg['is_fs'] = True
            session_dataset_cfg['fs_num'] = cfg.general_settings.fs_num

            if cfg.incremental_settings.lr_mul_fs is not None:
                _cfg.optimizer.lr = cfg.optimizer.lr * cfg.incremental_settings.lr_mul_fs
        else:
            raise ValueError("Unknown session type.")
        if cfg.incremental_settings.fix_fs:
            if session_type == 'fs':
                for param in model.backbone.parameters():
                    param.requires_grad = False
                model.backbone.norm_eval = True
                logger.info("Fix backbone.")
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = True
                model.backbone.norm_eval = False
        datasets = build_dataset(session_dataset_cfg)
        dist_check(torch.tensor(len(datasets), dtype=torch.float32, device=torch.cuda.current_device()))
        logger.info("The datasets are ensured to be with same size {}.".format(len(datasets)))
        total_datasets = copy.deepcopy(datasets)
        total_datasets = sum(datasets_memory, total_datasets)

        val_dataset_cfg = copy.deepcopy(cfg.data.val)
        val_dataset_cfg.cls_range = (base_cls + session_id * step, base_cls + (session_id + 1) * step)
        val_datasets.append(val_dataset_cfg)
        meta.update(
            dict(
                mmcls_version=__version__,
                config=cfg.pretty_text,
                CLASSES=datasets.CLASSES)
        )
        model.head.eval_classes = (0, base_cls + (session_id + 1) * step)
        logger.info("Model is evaluating {} to {}".format(*model.head.get_eval_classes()))
        if cfg.incremental_settings.vec_init:
            cls_means_inc = datasets.extract_prototype(cfg, model_inf=model)
            model.head.set_proto(
                cls_means_inc, start=base_cls + session_id * step, end=base_cls + (session_id + 1) * step
            )
            dist_check(model.head.etf_proto)
        if cfg.incremental_settings.use_lambda_lucir:
            lam = math.sqrt((base_cls + session_id * step) / base_cls)
            # lam = (base_cls + session_id * step) / base_cls
            logger.info("LUCIR: {} out of {} classes, lambda : {}, loss : {}".format(
                step, base_cls + session_id * step, lam, cfg.model.loss_dist.loss_weight * lam))
            model.dist_loss.loss_weight = cfg.model.loss_dist.loss_weight * lam
        meta['session_id'] = session_id + 1
        train_model(
            model,
            total_datasets,
            _cfg,
            val_dataset_cfg=val_datasets,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            device=cfg.device,
            meta=meta
        )
        logger.info("Getting exemplars.... , ")
        datasets.extract_exemplars(cfg, model)
        logger.info("...Done!, datasets is with {} len now.".format(len(datasets)))
        datasets_memory.append(datasets)
        if hasattr(model, 'save_pre_model'):
            logger.info("Saving previous model....")
            model.save_pre_model()

        if cfg.incremental_settings.two_stage:
            logger.info("Starting second stage...")
            from mmcil.models import ImageClassifierCILSecondStage
            model_extra = ImageClassifierCILSecondStage(
                prev_classifier=model,
            )
            datasets_extra = copy.deepcopy(datasets_memory[0])
            datasets_extra = sum(datasets_memory[1:], datasets_extra)
            train_model(
                model_extra,
                datasets_extra,
                copy.deepcopy(cfg),
                val_dataset_cfg=val_datasets,
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                device=cfg.device,
                meta=meta
            )


if __name__ == '__main__':
    main()
