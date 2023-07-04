import copy
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from mmcls.datasets import build_dataset
from mmcls.utils import wrap_distributed_model, wrap_non_distributed_model

from mmcil.datasets.utils import build_val_dataloader


def extract_features(cfg, model_inf, cls_start, cls_end):
    rank, world_size = get_dist_info()
    distributed = dist.is_available() and dist.is_initialized()
    val_dataset_cfg = copy.deepcopy(cfg.data.val)
    val_dataset_cfg.cls_range = (cls_start, cls_end)
    val_dataset = build_dataset(copy.deepcopy(val_dataset_cfg), dict(test_mode=True))
    val_loader = build_val_dataloader(cfg, val_dataset)
    memory = OrderedDict()
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model_inf = wrap_distributed_model(
            model_inf,
            cfg.device,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model_inf = wrap_non_distributed_model(
            model_inf, cfg.device, device_ids=cfg.gpu_ids)
    for cls_id in range(cls_start, cls_end):
        memory[cls_id] = []
    for data in val_loader:
        with torch.no_grad():
            result = model_inf(return_loss=False, return_feat=True, img=data['img'], gt_label=None)
        for idx, cur in enumerate(data['img_metas'].data[0]):
            cls_id = cur['cls_id']
            img_id = cur['img_id']
            memory[cls_id].append((img_id, result[idx].to(device='cpu')))
    if rank == 0:
        print()

    if distributed:
        dist.barrier(device_ids=[torch.cuda.current_device()])
        for cls in sorted(memory.keys()):
            memory_cls = memory[cls]
            recv_list = [None for _ in range(world_size)]
            # gather all result part
            dist.all_gather_object(recv_list, memory_cls)
            memory_cls = []
            for itm in recv_list:
                memory_cls.extend(itm)
            memory_cls.sort(key=lambda x: x[0])
            memory[cls] = torch.stack(list(map(lambda x: x[1], memory_cls)))
    else:
        for cls in sorted(memory.keys()):
            memory_cls = memory[cls]
            memory_cls.sort(key=lambda x: x[0])
            memory[cls] = torch.stack(list(map(lambda x: x[1], memory_cls)))

    return memory
