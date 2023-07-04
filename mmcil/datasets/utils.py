from mmcv.runner import get_dist_info
from torch import distributed as dist
from mmcls.datasets import build_dataloader


def build_val_dataloader(cfg, dataset):
    rank, world_size = get_dist_info()
    distributed = dist.is_available() and dist.is_initialized()
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=world_size if distributed else 1,
        dist=distributed,
        round_up=False,
        seed=0,  # usually, val dataset does not need random
        shuffle=False,
        sampler_cfg=None,
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
    val_loader_cfg = {**loader_cfg, **cfg.data.get('val_dataloader', {})}

    data_loader = build_dataloader(dataset, **val_loader_cfg)
    return data_loader
