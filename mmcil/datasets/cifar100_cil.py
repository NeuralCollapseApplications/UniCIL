import copy
import numpy as np
import os.path as osp
import pickle
from collections import defaultdict, OrderedDict
from typing import List, Dict, Mapping, Tuple

import torch
from typing_extensions import Literal

from torch.utils.data import Dataset
import torch.distributed as dist

from mmcv.runner import get_dist_info

from mmcls.datasets.utils import download_and_extract_archive, check_integrity
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.pipelines import Compose
from mmcls.utils import get_root_logger, wrap_distributed_model, wrap_non_distributed_model

from mmcil.datasets.utils import build_val_dataloader

CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',

    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',
    'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Please refer to the following:
# https://github.com/Yujun-Shi/CwD/blob/c085e56550e4a60602168ec22361295cab94da36/src/datasets/dataset_config.py#L37
CLASS_ORDER = [
    68, 56, 78, 8, 23, 84, 90, 65, 74, 76,
    40, 89, 3, 92, 55, 9, 26, 80, 43, 38,
    58, 70, 77, 1, 85, 19, 17, 50, 28, 53,
    13, 81, 45, 82, 6, 59, 83, 16, 15, 44,
    91, 41, 72, 60, 79, 52, 20, 10, 31, 54,
    37, 95, 14, 71, 96, 98, 97, 2, 64, 66,
    42, 22, 35, 86, 24, 34, 87, 21, 99, 0,
    88, 27, 18, 94, 11, 12, 47, 25, 30, 46,
    62, 69, 36, 61, 7, 63, 75, 5, 32, 4,
    51, 48, 73, 93, 39, 67, 29, 49, 57, 33
]

ORDER_2023 = [
    56, 12, 68, 0, 82, 66, 91, 44, 46, 20, 35, 76, 83, 9, 16, 89, 26, 24, 2, 43,
    84, 96, 18, 21, 8, 4, 61, 37, 95, 30, 14, 50, 81, 6, 57, 64, 10, 85, 42, 41,
    19, 5, 31, 1, 11, 59, 36, 97, 40, 60, 79, 23, 67, 51, 62, 27, 54, 78, 13, 72,
    34, 98, 94, 45, 7, 80, 90, 65, 99, 15, 38, 88, 73, 74, 75, 49, 29, 32, 48, 63,
    71, 28, 58, 93, 69, 39, 77, 47, 53, 17, 22, 86, 52, 3, 92, 33, 55, 70, 25, 87,
]


@DATASETS.register_module()
class CIFAR100CILDataset(Dataset):
    BASE_LT = 500
    """CIRFAR100 dataset for few shot class-incremental classification.
    few_cls is None when performing usual training, is tuple for few-shot training
    """

    # Copy and paste from torchvision
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(
            self,
            data_prefix: str,
            pipeline: List[Dict],
            cls_range: Tuple[int, int] = (0, 100),
            subset: Literal['train', 'test'] = 'train',
            test_mode: bool = False,
            order: List[int] = None,
            is_lt: bool = False,
            lt_factor: int = 0.,
            lt_shuffle: bool = False,
    ):
        rank, world_size = get_dist_info()
        self.test_mode = test_mode

        self.logger = get_root_logger()
        assert not cls_range == (0, 0)
        if self.test_mode:
            self.logger.info("Validation set : {} to {}".format(*cls_range))
        else:
            self.logger.info("Training set : {} to {}".format(*cls_range))
        self.cls_range = cls_range

        if order is None:
            self.CLASS_ORDER = CLASS_ORDER
        else:
            self.CLASS_ORDER = copy.deepcopy(order)

        if not test_mode:
            # No need to print order in the test mode.
            self.logger.info("order : {}".format(self.CLASS_ORDER))

        self.data_prefix = data_prefix
        assert isinstance(pipeline, list), 'pipeline is type of list'
        self.pipeline = Compose(pipeline)

        if rank == 0 and not self._check_integrity():
            download_and_extract_archive(
                self.url,
                self.data_prefix,
                filename=self.filename,
                md5=self.tgz_md5)

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. ' \
                f'Please download the dataset manually through {self.url}.'

        self.subset = subset
        if self.subset == 'train':
            downloaded_list = self.train_list
        elif self.subset == 'test':
            downloaded_list = self.test_list
        else:
            raise NotImplementedError

        self.CLASSES = self.get_classes(cls_range[0], cls_range[1])

        self.data_infos = self.load_annotations(downloaded_list)
        self.data_infos = sorted(self.data_infos, key=lambda x: (x['cls_id'], ['img_id']))

        if is_lt:
            self.log_class_num()
            assert not test_mode
            img_num_per_cls = self.get_img_num_per_cls(len(ORDER_2023), imb_factor=lt_factor)
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
        for idx in range(len(ORDER_2023)):
            cls_cnt_list.append(cls_cnt[idx])
        self.logger.info("Dataset len : {}".format(len(self)))
        self.logger.info("{}".format(cls_cnt_list))

    def get_eval_classes(self):
        return self.cls_range

    def get_classes(self, a, b):
        return [CLASSES[idx] for idx in self.CLASS_ORDER][a:b]

    @property
    def class_to_idx(self) -> Mapping:
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(CLASSES)}

    def load_annotations(self, downloaded_list) -> List:
        """Load annotation according to the classes subset."""
        imgs = []
        gt_labels = []

        # load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = osp.join(self.data_prefix, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                imgs.append(entry['data'])
                if 'labels' in entry:
                    gt_labels.extend(entry['labels'])
                else:
                    gt_labels.extend(entry['fine_labels'])

        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)
        imgs = imgs.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        data_infos = []

        cls_cnt = defaultdict(lambda: 0)
        for img, _gt_label in zip(imgs, gt_labels):
            if CLASSES[_gt_label] in self.CLASSES:
                gt_label_cur = self.CLASS_ORDER.index(_gt_label)
                gt_label = np.array(gt_label_cur, dtype=np.int64)
                info = {'img': img, 'gt_label': gt_label, 'cls_id': gt_label_cur, 'img_id': cls_cnt[gt_label_cur]}
                cls_cnt[gt_label_cur] += 1
                data_infos.append(info)
        return data_infos

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict:
        if self.test_mode:
            return self.prepare_data(idx)
        else:
            return self.prepare_data(idx)

    def prepare_data(self, idx: int) -> Dict:
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    # from mmcls, thx
    def _load_meta(self):
        path = osp.join(self.data_prefix, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError(
                'Dataset metadata file not found or corrupted.' +
                ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            for idx, name in enumerate(data[self.meta['key']]):
                assert CLASSES[idx] == name

    # from mmcls, thx
    def _check_integrity(self):
        root = self.data_prefix
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = osp.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

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
