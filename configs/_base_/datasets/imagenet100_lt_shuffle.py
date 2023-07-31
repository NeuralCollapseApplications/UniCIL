_base_ = ['./rand_aug.py']

dataset_type = 'ImagenetCIL100'
img_size = 224
_img_resize_size = 256
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
meta_keys = ('filename', 'ori_filename', 'ori_shape',
             'img_shape', 'flip', 'flip_direction',
             'img_norm_cfg', 'cls_id', 'img_id')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=img_size,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'], meta_keys=meta_keys)
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(_img_resize_size, -1),
        backend='pillow',
        interpolation='bicubic'),
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
        type=dataset_type,
        data_prefix='data/imagenet/train',
        pipeline=train_pipeline,
        cls_range=(0, 50),
        is_lt=True,
        lt_factor=0.01,
        lt_shuffle=True,
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/imagenet/_val',
        pipeline=test_pipeline,
        cls_range=(0, 0),
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/imagenet/_val',
        pipeline=test_pipeline,
        cls_range=(0, 0),
    )
)
