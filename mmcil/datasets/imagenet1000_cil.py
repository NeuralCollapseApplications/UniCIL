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

ORDER = [
    54, 7, 894, 512, 126, 337, 988, 11, 284, 493, 133, 783, 192, 979, 622, 215, 240, 548, 238, 419,
    274, 108, 928, 856, 494, 836, 473, 650, 85, 262, 508, 590, 390, 174, 637, 288, 658, 219, 912, 142,
    852, 160, 704, 289, 123, 323, 600, 542, 999, 634, 391, 761, 490, 842, 127, 850, 665, 990, 597, 722,
    748, 14, 77, 437, 394, 859, 279, 539, 75, 466, 886, 312, 303, 62, 966, 413, 959, 782, 509, 400,
    471, 632, 275, 730, 105, 523, 224, 186, 478, 507, 470, 906, 699, 989, 324, 812, 260, 911, 446, 44,
    765, 759, 67, 36, 5, 30, 184, 797, 159, 741, 954, 465, 533, 585, 150, 101, 897, 363, 818, 620,
    824, 154, 956, 176, 588, 986, 172, 223, 461, 94, 141, 621, 659, 360, 136, 578, 163, 427, 70, 226,
    925, 596, 336, 412, 731, 755, 381, 810, 69, 898, 310, 120, 752, 93, 39, 326, 537, 905, 448, 347,
    51, 615, 601, 229, 947, 348, 220, 949, 972, 73, 913, 522, 193, 753, 921, 257, 957, 691, 155, 820,
    584, 948, 92, 582, 89, 379, 392, 64, 904, 169, 216, 694, 103, 410, 374, 515, 484, 624, 409, 156,
    455, 846, 344, 371, 468, 844, 276, 740, 562, 503, 831, 516, 663, 630, 763, 456, 179, 996, 936, 248,
    333, 941, 63, 738, 802, 372, 828, 74, 540, 299, 750, 335, 177, 822, 643, 593, 800, 459, 580, 933,
    306, 378, 76, 227, 426, 403, 322, 321, 808, 393, 27, 200, 764, 651, 244, 479, 3, 415, 23, 964,
    671, 195, 569, 917, 611, 644, 707, 355, 855, 8, 534, 657, 571, 811, 681, 543, 313, 129, 978, 592,
    573, 128, 243, 520, 887, 892, 696, 26, 551, 168, 71, 398, 778, 529, 526, 792, 868, 266, 443, 24,
    57, 15, 871, 678, 745, 845, 208, 188, 674, 175, 406, 421, 833, 106, 994, 815, 581, 676, 49, 619,
    217, 631, 934, 932, 568, 353, 863, 827, 425, 420, 99, 823, 113, 974, 438, 874, 343, 118, 340, 472,
    552, 937, 0, 10, 675, 316, 879, 561, 387, 726, 255, 407, 56, 927, 655, 809, 839, 640, 297, 34,
    497, 210, 606, 971, 589, 138, 263, 587, 993, 973, 382, 572, 735, 535, 139, 524, 314, 463, 895, 376,
    939, 157, 858, 457, 935, 183, 114, 903, 767, 666, 22, 525, 902, 233, 250, 825, 79, 843, 221, 214,
    205, 166, 431, 860, 292, 976, 739, 899, 475, 242, 961, 531, 110, 769, 55, 701, 532, 586, 729, 253,
    486, 787, 774, 165, 627, 32, 291, 962, 922, 222, 705, 454, 356, 445, 746, 776, 404, 950, 241, 452,
    245, 487, 706, 2, 137, 6, 98, 647, 50, 91, 202, 556, 38, 68, 649, 258, 345, 361, 464, 514,
    958, 504, 826, 668, 880, 28, 920, 918, 339, 315, 320, 768, 201, 733, 575, 781, 864, 617, 171, 795,
    132, 145, 368, 147, 327, 713, 688, 848, 690, 975, 354, 853, 148, 648, 300, 436, 780, 693, 682, 246,
    449, 492, 162, 97, 59, 357, 198, 519, 90, 236, 375, 359, 230, 476, 784, 117, 940, 396, 849, 102,
    122, 282, 181, 130, 467, 88, 271, 793, 151, 847, 914, 42, 834, 521, 121, 29, 806, 607, 510, 837,
    301, 669, 78, 256, 474, 840, 52, 505, 547, 641, 987, 801, 629, 491, 605, 112, 429, 401, 742, 528,
    87, 442, 910, 638, 785, 264, 711, 369, 428, 805, 744, 380, 725, 480, 318, 997, 153, 384, 252, 985,
    538, 654, 388, 100, 432, 832, 565, 908, 367, 591, 294, 272, 231, 213, 196, 743, 817, 433, 328, 970,
    969, 4, 613, 182, 685, 724, 915, 311, 931, 865, 86, 119, 203, 268, 718, 317, 926, 269, 161, 209,
    807, 645, 513, 261, 518, 305, 758, 872, 58, 65, 146, 395, 481, 747, 41, 283, 204, 564, 185, 777,
    33, 500, 609, 286, 567, 80, 228, 683, 757, 942, 134, 673, 616, 960, 450, 350, 544, 830, 736, 170,
    679, 838, 819, 485, 430, 190, 566, 511, 482, 232, 527, 411, 560, 281, 342, 614, 662, 47, 771, 861,
    692, 686, 277, 373, 16, 946, 265, 35, 9, 884, 909, 610, 358, 18, 737, 977, 677, 803, 595, 135,
    458, 12, 46, 418, 599, 187, 107, 992, 770, 298, 104, 351, 893, 698, 929, 502, 273, 20, 96, 791,
    636, 708, 267, 867, 772, 604, 618, 346, 330, 554, 816, 664, 716, 189, 31, 721, 712, 397, 43, 943,
    804, 296, 109, 576, 869, 955, 17, 506, 963, 786, 720, 628, 779, 982, 633, 891, 734, 980, 386, 365,
    794, 325, 841, 878, 370, 695, 293, 951, 66, 594, 717, 116, 488, 796, 983, 646, 499, 53, 1, 603,
    45, 424, 875, 254, 237, 199, 414, 307, 362, 557, 866, 341, 19, 965, 143, 555, 687, 235, 790, 125,
    173, 364, 882, 727, 728, 563, 495, 21, 558, 709, 719, 877, 352, 83, 998, 991, 469, 967, 760, 498,
    814, 612, 715, 290, 72, 131, 259, 441, 924, 773, 48, 625, 501, 440, 82, 684, 862, 574, 309, 408,
    680, 623, 439, 180, 652, 968, 889, 334, 61, 766, 399, 598, 798, 653, 930, 149, 249, 890, 308, 881,
    40, 835, 577, 422, 703, 813, 857, 995, 602, 583, 167, 670, 212, 751, 496, 608, 84, 639, 579, 178,
    489, 37, 197, 789, 530, 111, 876, 570, 700, 444, 287, 366, 883, 385, 536, 460, 851, 81, 144, 60,
    251, 13, 953, 270, 944, 319, 885, 710, 952, 517, 278, 656, 919, 377, 550, 207, 660, 984, 447, 553,
    338, 234, 383, 749, 916, 626, 462, 788, 434, 714, 799, 821, 477, 549, 661, 206, 667, 541, 642, 689,
    194, 152, 981, 938, 854, 483, 332, 280, 546, 389, 405, 545, 239, 896, 672, 923, 402, 423, 907, 888,
    140, 870, 559, 756, 25, 211, 158, 723, 635, 302, 702, 453, 218, 164, 829, 247, 775, 191, 732, 115,
    331, 901, 416, 873, 754, 900, 435, 762, 124, 304, 329, 349, 295, 95, 451, 285, 225, 945, 697, 417,
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
class ImagenetCIL1000(ImageNet):
    def __init__(
            self,
            data_prefix: str,
            pipeline: Sequence = (),
            classes: Union[str, Sequence[str], None] = None,
            ann_file: Optional[str] = None,
            test_mode: bool = False,
            cls_range: Tuple[int, int] = (0, 1000),
            file_client_args: Optional[dict] = None
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
        self.logger.info("[{}]: datasets prepared.".format(self.__class__.__name__))

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
