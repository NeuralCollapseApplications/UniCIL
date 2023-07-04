import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def dist_check(dist_tensor: torch.Tensor):
    rank, world_size = get_dist_info()
    if world_size > 1:
        dist_tensor_min = dist_tensor.detach().clone()
        dist_tensor_max = dist_tensor.detach().clone()
        dist.all_reduce(dist_tensor_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(dist_tensor_max, op=dist.ReduceOp.MAX)
        assert torch.allclose(dist_tensor_min, dist_tensor_max)
