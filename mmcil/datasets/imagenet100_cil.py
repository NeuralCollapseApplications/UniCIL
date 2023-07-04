import copy
from collections import defaultdict, OrderedDict
from typing import Sequence, Union, Optional, Tuple, List, Dict, Callable

import numpy as np
import torch
import torch.distributed as dist
from mmcv import FileClient
from mmcv.runner import get_dist_info

from mmcil.datasets.utils import build_val_dataloader
from mmcls.datasets import ImageNet, DATASETS
from mmcls.datasets.pipelines import Compose
from mmcls.utils import get_root_logger, wrap_non_distributed_model, wrap_distributed_model

SELECT = sorted([
    54, 7, 894, 512, 126, 337, 988, 11, 284, 493, 133, 783, 192, 979, 622, 215, 240, 548, 238, 419,
    274, 108, 928, 856, 494, 836, 473, 650, 85, 262, 508, 590, 390, 174, 637, 288, 658, 219, 912, 142,
    852, 160, 704, 289, 123, 323, 600, 542, 999, 634, 391, 761, 490, 842, 127, 850, 665, 990, 597, 722,
    748, 14, 77, 437, 394, 859, 279, 539, 75, 466, 886, 312, 303, 62, 966, 413, 959, 782, 509, 400,
    471, 632, 275, 730, 105, 523, 224, 186, 478, 507, 470, 906, 699, 989, 324, 812, 260, 911, 446, 44,
])

ORDER_1993 = [
    68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38,
    58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44,
    91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66,
    42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46,
    62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33,
]

ORDER = [SELECT[idx] for idx in ORDER_1993]

ORDER_2023 = [
    56, 12, 68, 0, 82, 66, 91, 44, 46, 20, 35, 76, 83, 9, 16, 89, 26, 24, 2, 43,
    84, 96, 18, 21, 8, 4, 61, 37, 95, 30, 14, 50, 81, 6, 57, 64, 10, 85, 42, 41,
    19, 5, 31, 1, 11, 59, 36, 97, 40, 60, 79, 23, 67, 51, 62, 27, 54, 78, 13, 72,
    34, 98, 94, 45, 7, 80, 90, 65, 99, 15, 38, 88, 73, 74, 75, 49, 29, 32, 48, 63,
    71, 28, 58, 93, 69, 39, 77, 47, 53, 17, 22, 86, 52, 3, 92, 33, 55, 70, 25, 87,
]


def find_folders_cil(root: str, file_client: FileClient) -> Tuple[List[str], Dict[str, int]]:
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        Tuple[List[str], Dict[str, int]]:

        - folders: The name of sub folders under the root.
        - folder_to_idx: The map from folder name to class idx.
    """
    folders = list(
        file_client.list_dir_or_file(
            root,
            list_dir=True,
            list_file=False,
            recursive=False,
        ))
    folders.sort()
    folders = [folders[idx] for idx in ORDER]
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folders, folder_to_idx


def get_samples(root: str, folder_to_idx: Dict[str, int], is_valid_file: Callable, file_client: FileClient):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        is_valid_file (Callable): A function that takes path of a file
            and check if the file is a valid sample file.

    Returns:
        Tuple[list, set]:

        - samples: a list of tuple where each element is (image, class_idx)
        - empty_folders: The folders don't have any valid files.
    """
    samples = []
    available_classes = set()

    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = file_client.join_path(root, folder_name)
        files = list(
            file_client.list_dir_or_file(
                _dir,
                list_dir=False,
                list_file=True,
                recursive=True,
            ))
        for file in sorted(list(files)):
            if is_valid_file(file):
                path = file_client.join_path(folder_name, file)
                item = (path, folder_to_idx[folder_name])
                samples.append(item)
                available_classes.add(folder_name)

    empty_folders = set(folder_to_idx.keys()) - available_classes

    return samples, empty_folders


@DATASETS.register_module()
class ImagenetCIL100(ImageNet):
    BASE_LT = 1300

    def __init__(
            self,
            data_prefix: str,
            pipeline: Sequence = (),
            classes: Union[str, Sequence[str], None] = None,
            ann_file: Optional[str] = None,
            test_mode: bool = False,
            cls_range: Tuple[int, int] = (0, 100),
            file_client_args: Optional[dict] = None,
            is_lt: bool = False,
            lt_factor: int = 0.,
            lt_shuffle: bool = False,
    ):
        assert ann_file is None
        self.cls_range = cls_range
        self.logger = get_root_logger()
        super().__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            ann_file=ann_file,
            test_mode=test_mode,
            file_client_args=file_client_args
        )
        self.data_infos = sorted(self.data_infos, key=lambda x: (x['cls_id'], ['img_id']))
        if is_lt:
            self.log_class_num()
            assert not test_mode
            img_num_per_cls = self.get_img_num_per_cls(len(self.CLASSES), imb_factor=lt_factor)
            if lt_shuffle:
                self.logger.info("LT shuffle Enabled")
                order = ORDER_2023
                img_num_per_cls_new = []
                for idx in range(len(order)):
                    img_num_per_cls_new.append(img_num_per_cls[order[idx]])
                img_num_per_cls = img_num_per_cls_new
            self.gen_imbalanced_data(img_num_per_cls)
            self.log_class_num()
        self.logger.info("[{}]: datasets prepared.".format(self.__class__.__name__))

    # LT codes
    def get_img_num_per_cls(self, cls_num, imb_type='exp', imb_factor=0.01):
        img_max = self.BASE_LT
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    # LT codes
    def gen_imbalanced_data(self, img_num_per_cls):
        data_infos = self.data_infos
        self.data_infos = []
        cls_cnt = defaultdict(lambda: 0)
        for data_info in data_infos:
            label = data_info['cls_id']
            if cls_cnt[label] >= img_num_per_cls[label]:
                continue
            else:
                cls_cnt[label] += 1
                self.data_infos.append(data_info)

    def log_class_num(self):
        cls_cnt = defaultdict(lambda: 0)
        cls_cnt_list = []
        for data_info in self.data_infos:
            label = data_info['cls_id']
            cls_cnt[label] += 1
        for idx in range(len(self.CLASSES)):
            cls_cnt_list.append(cls_cnt[idx])
        self.logger.info("Dataset len : {}".format(len(self)))
        self.logger.info("{}".format(cls_cnt_list))

    def load_annotations(self):
        """Load image paths and gt_labels."""
        samples = self._find_samples()

        data_infos = []
        cls_cnt = defaultdict(lambda: 0)
        for filename, gt_label in samples:
            if self.cls_range[0] <= gt_label < self.cls_range[1]:
                info = {
                    'img_prefix': self.data_prefix,
                    'img_info': {'filename': filename},
                    'gt_label': np.array(gt_label, dtype=np.int64),
                    'cls_id': gt_label,
                    'img_id': cls_cnt[gt_label]
                }
                cls_cnt[gt_label] += 1
                data_infos.append(info)
        str_print = "[{}]: ".format(self.__class__.__name__)
        for cls in sorted(cls_cnt):
            str_print += "{} : {} ".format(cls, cls_cnt[cls])
        self.logger.info(str_print)

        return data_infos

    def _find_samples(self):
        """find samples from ``data_prefix``."""
        file_client = FileClient.infer_client(self.file_client_args,
                                              self.data_prefix)
        classes, folder_to_idx = find_folders_cil(self.data_prefix, file_client)
        samples, empty_classes = get_samples(
            self.data_prefix,
            folder_to_idx,
            is_valid_file=self.is_valid_file,
            file_client=file_client,
        )

        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        self.CLASSES = classes

        if empty_classes:
            raise ValueError(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}")

        self.folder_to_idx = folder_to_idx

        return samples

    def get_eval_classes(self):
        return self.cls_range

    def extract_exemplars(self, cfg, model_inf, num_samples_per_cls=20):
        rank, world_size = get_dist_info()
        distributed = dist.is_available() and dist.is_initialized()
        old_pipline = self.pipeline
        self.pipeline = Compose(copy.deepcopy(cfg.data.val.pipeline))
        exemplars_loader = build_val_dataloader(cfg, self)
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
        for cls_id in range(*self.cls_range):
            memory[cls_id] = []
        for data in exemplars_loader:
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

        data_infos = []
        selected = {}
        for cls in sorted(memory.keys()):
            cls_mean = memory[cls].mean(dim=0, keepdim=False)
            cls_mean = cls_mean / torch.norm(cls_mean, p=2)
            feat_dist = memory[cls] @ cls_mean
            max_indices = feat_dist.argsort(descending=True)[:num_samples_per_cls]
            selected[cls] = max_indices.tolist()

        for data_info in self.data_infos:
            cls_id = data_info['cls_id']
            img_id = data_info['img_id']
            if img_id in selected[cls_id]:
                data_infos.append(data_info)

        self.data_infos = data_infos
        self.pipeline = old_pipline

    def extract_prototype(self, cfg, model_inf):
        rank, world_size = get_dist_info()
        distributed = dist.is_available() and dist.is_initialized()
        old_pipline = self.pipeline
        self.pipeline = Compose(copy.deepcopy(cfg.data.val.pipeline))
        exemplars_loader = build_val_dataloader(cfg, self)
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
        for cls_id in range(*self.cls_range):
            memory[cls_id] = []
        for data in exemplars_loader:
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

        results = []
        for cls in sorted(memory.keys()):
            cls_mean = memory[cls].mean(dim=0, keepdim=False)
            cls_mean = cls_mean / torch.norm(cls_mean, p=2)
            results.append(cls_mean)

        self.pipeline = old_pipline

        return results
