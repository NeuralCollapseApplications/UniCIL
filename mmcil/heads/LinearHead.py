import torch

from mmcls.models import HEADS, LinearClsHead


@HEADS.register_module()
class LinearCILClsHead(LinearClsHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_classes = (0, self.num_classes)
        self.use_proto = None

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        assert len(x.shape) == 2
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def get_eval_classes(self):
        return self.eval_classes

    def simple_test(self, x, softmax=False, post_process=False, return_feat=False):
        if return_feat:
            return self.pre_logits(x)
        else:
            return super().simple_test(x, softmax=softmax, post_process=post_process)
