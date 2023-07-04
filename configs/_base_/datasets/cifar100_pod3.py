img_size = 32
_img_resize_size = 36
img_norm_cfg = dict(mean=[129.304, 124.070, 112.434], std=[68.170, 65.392, 70.418], to_rgb=False)
meta_keys = ('filename', 'ori_filename', 'ori_shape',
             'img_shape', 'flip', 'flip_direction',
             'img_norm_cfg', 'cls_id', 'img_id')

train_pipeline = [
    dict(type='RandomResizedCrop', size=img_size, scale=(0.6, 1.), interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

test_pipeline = [
    dict(type='Resize', size=(_img_resize_size, -1), interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train_dataloader=dict(
        persistent_workers=True,
    ),
    val_dataloader=dict(
        persistent_workers=False,
    ),
    test_dataloader=dict(
        persistent_workers=False,
    ),
    train=dict(
        type='CIFAR100CILDataset',
        data_prefix='/opt/data/cifar',
        pipeline=train_pipeline,
        cls_range=(0, 50),
        subset='train',
        order=[71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36, 90, 58, 21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64, 18, 60, 50, 63, 61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97, 75, 2, 17, 93, 33, 84, 99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11]
    ),
    val=dict(
        type='CIFAR100CILDataset',
        data_prefix='/opt/data/cifar',
        pipeline=test_pipeline,
        cls_range=(0, 0),
        subset='test',
        order=[71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36, 90, 58, 21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64, 18, 60, 50, 63, 61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97, 75, 2, 17, 93, 33, 84, 99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11]
    ),
    test=dict(
        type='CIFAR100CILDataset',
        data_prefix='/opt/data/cifar',
        pipeline=test_pipeline,
        cls_range=(0, 0),
        subset='test',
        order=[71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36, 90, 58, 21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64, 18, 60, 50, 63, 61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97, 75, 2, 17, 93, 33, 84, 99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11]
    )
)
