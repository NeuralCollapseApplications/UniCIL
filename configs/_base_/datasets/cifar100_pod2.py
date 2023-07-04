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
        order=[58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70, 90, 63, 67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88, 95, 85, 4, 60, 36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97, 82, 98, 26, 47, 44, 62, 13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61]
    ),
    val=dict(
        type='CIFAR100CILDataset',
        data_prefix='/opt/data/cifar',
        pipeline=test_pipeline,
        cls_range=(0, 0),
        subset='test',
        order=[58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70, 90, 63, 67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88, 95, 85, 4, 60, 36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97, 82, 98, 26, 47, 44, 62, 13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61]
    ),
    test=dict(
        type='CIFAR100CILDataset',
        data_prefix='/opt/data/cifar',
        pipeline=test_pipeline,
        cls_range=(0, 0),
        subset='test',
        order=[58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70, 90, 63, 67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88, 95, 85, 4, 60, 36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97, 82, 98, 26, 47, 44, 62, 13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61]
    )
)
